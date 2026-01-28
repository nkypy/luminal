use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use cube::op::{EgglogOp, HLIROps, Input, Output};

// 要测试的函数
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn fib_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

fn op_dyn_trait_benchmark(c: &mut Criterion) {
    const SIZE: usize = 512;
    let mut ops: Vec<Box<dyn EgglogOp>> = Vec::with_capacity(SIZE * 2);
    for i in 0..SIZE {
        ops.push(Box::new(Input {
            node: i,
            label: "example".to_string(),
        }));
        ops.push(Box::new(Output { node: i }));
    }
    c.bench_with_input(BenchmarkId::new("op_dyn_trait", SIZE), &ops, |b, ops| {
        b.iter(|| {
            for op in ops {
                op.name();
            }
        })
    });
}

fn op_enum_dispatch_benchmark(c: &mut Criterion) {
    const SIZE: usize = 512;
    let mut ops: Vec<HLIROps> = Vec::with_capacity(SIZE * 2);
    for i in 0..SIZE {
        ops.push(
            Input {
                node: i,
                label: "example".to_string(),
            }
            .into(),
        );
        ops.push(Output { node: i }.into());
    }
    c.bench_with_input(
        BenchmarkId::new("op_enum_dispatch", SIZE),
        &ops,
        |b, ops| {
            b.iter(|| {
                for op in ops {
                    op.name();
                }
            })
        },
    );
}

criterion_group!(
    benches,
    fib_benchmark,
    op_dyn_trait_benchmark,
    op_enum_dispatch_benchmark
);
criterion_main!(benches);
