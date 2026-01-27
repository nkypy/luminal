use std::marker::PhantomData;

use cubecl::{
    benchmark::Benchmark, future, prelude::*, profile::TimingMethod, server::Handle,
    std::tensor::compact_strides,
};

/// Simple GpuTensor
#[derive(Debug)]
pub struct GpuTensor<R: Runtime, F: Float + CubeElement> {
    data: Handle,
    pub shape: Vec<usize>,
    strides: Vec<usize>,
    _r: PhantomData<R>,
    _f: PhantomData<F>,
}

impl<R: Runtime, F: Float + CubeElement> Clone for GpuTensor<R, F> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(), // Handle is a pointer to the data, so cloning it is cheap
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _r: PhantomData,
            _f: PhantomData,
        }
    }
}

impl<R: Runtime, F: Float + CubeElement> GpuTensor<R, F> {
    /// Create a GpuTensor with a shape filled by number in order
    pub fn arange(shape: Vec<usize>, client: &ComputeClient<R>) -> Self {
        let size = shape.iter().product();
        let data: Vec<F> = (0..size).map(|i| F::from_int(i as i64)).collect();
        let data = client.create_from_slice(F::as_bytes(&data));

        let strides = compact_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        }
    }

    /// Create an empty GpuTensor with a shape
    pub fn empty(shape: Vec<usize>, client: &ComputeClient<R>) -> Self {
        let size = shape.iter().product::<usize>() * core::mem::size_of::<F>();
        let data = client.empty(size);

        let strides = compact_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        }
    }

    /// Create a TensorArg to pass to a kernel
    pub fn into_tensor_arg(&self, line_size: usize) -> TensorArg<'_, R> {
        unsafe { TensorArg::from_raw_parts::<F>(&self.data, &self.strides, &self.shape, line_size) }
    }

    /// Return the data from the client
    pub fn read(self, client: &ComputeClient<R>) -> Vec<F> {
        let bytes = client.read_one(self.data);
        F::from_bytes(&bytes).to_vec()
    }
}

#[cube(launch_unchecked)]
fn reduce_matrix<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let mut acc = Line::new(F::new(0.0f32));
    for i in 0..input.shape(2) / LINE_SIZE {
        acc += input
            [CUBE_POS_X as usize * input.stride(0) + UNIT_POS_X as usize * input.stride(1) + i];
    }
    output[CUBE_POS_X as usize * output.stride(0) + UNIT_POS_X as usize] = acc;
}

pub fn launch<R: Runtime, F: Float + CubeElement>(device: &R::Device) {
    let client = R::client(device);

    let bench1 = ReductionBench::<R, F> {
        input_shape: vec![64, 256, 1024],
        client: client.clone(),
        _f: PhantomData,
    };
    let bench2 = ReductionBench::<R, F> {
        input_shape: vec![64, 64, 4096],
        client: client.clone(),
        _f: PhantomData,
    };

    for bench in [bench1, bench2] {
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::System).unwrap());
    }
}

fn main() {
    #[cfg(feature = "wgpu")]
    launch::<cube::WgpuRuntime, f32>(&Default::default());
    #[cfg(feature = "cuda")]
    launch::<cube::CudaRuntime, f32>(&Default::default());
    #[cfg(feature = "cpu")]
    launch::<cube::CpuRuntime, f32>(&Default::default());
}

pub struct ReductionBench<R: Runtime, F: Float + CubeElement> {
    input_shape: Vec<usize>,
    client: ComputeClient<R>,
    _f: PhantomData<F>,
}

const LINE_SIZE: usize = 4;

impl<R: Runtime, F: Float + CubeElement> Benchmark for ReductionBench<R, F> {
    type Input = GpuTensor<R, F>;
    type Output = GpuTensor<R, F>;

    fn prepare(&self) -> Self::Input {
        GpuTensor::<R, F>::arange(self.input_shape.clone(), &self.client)
    }

    fn name(&self) -> String {
        format!("{}-reduction-{:?}", R::name(&self.client), self.input_shape).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).expect("Failed to sync");
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let output_shape: Vec<usize> = vec![self.input_shape[0]];
        let output = GpuTensor::<R, F>::empty(output_shape, &self.client);

        unsafe {
            reduce_matrix::launch_unchecked::<F, R>(
                &self.client,
                CubeCount::Static(self.input_shape[0] as u32, 1, 1),
                CubeDim::new_3d(self.input_shape[1] as u32, 1, 1),
                input.into_tensor_arg(LINE_SIZE),
                output.into_tensor_arg(LINE_SIZE),
            )
            .unwrap();
        }

        Ok(output)
    }
}
