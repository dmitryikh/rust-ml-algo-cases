extern crate ml_algo;
extern crate kdtree;

use kdtree::KdTree;
use kdtree::distance::squared_euclidean;

use ml_algo::matrix::DMatrix;
use ml_algo::mshift::{MeanShift, MeanShiftOptions, SeedOptions};


#[test]
fn carnival_cruise() {
    let filename = "data/carnival_cruise_checkins.dat";
    let train: DMatrix<f64> = DMatrix::from_csv(filename, 1, ',', Some(&[0, 1])).unwrap();
    let offices: DMatrix<f64> = DMatrix::from_csv("data/carnival_cruise_offices.csv", 1, ',', Some(&[0, 1])).unwrap();
    let mut ms = MeanShift::new( MeanShiftOptions::new()
                                       .bandwidth(0.1)
                                       .seed(SeedOptions::Bins(0.1, 1))
                               );
    ms.fit(&train).unwrap();
    println!("n_clusters = \n{:?}", ms.n_clusters());
    println!("centers = \n{}", ms.centers());

    // Ищем для каждого кластера ближайший офис
    let mut kdtree = KdTree::new_with_capacity(offices.cols(), offices.rows());
    (0..offices.rows()).for_each(|i| kdtree.add(offices.get_row(i), i).unwrap());
    let mut min_dist = Vec::with_capacity(ms.n_clusters());
    for i in 0..ms.n_clusters() {
        kdtree.nearest(ms.centers().get_row(i), 1, &squared_euclidean).unwrap()
            .iter().take(1).for_each(|&(d, _)| min_dist.push(d.sqrt()));
    }
    // сортируем по возрастанию расстояния до офиса
    let mut ids = (0..ms.n_clusters()).collect::<Vec<usize>>();
    ids.sort_unstable_by(|&a, &b| min_dist[a].partial_cmp(&min_dist[b]).unwrap());

    // Печатаем 20 ближайших кластеров
    println!("Nearest clusters:");
    for i in 0..20 {
        println!("Cluster {}: dist {}, coord {:?}", i + 1, min_dist[ids[i]], ms.centers().get_row(ids[i]));
    }

    // Проверяем координаты ближайшего
    let closest = ms.centers().get_row(ids[0]);
    assert!((closest[0] - (-33.866)).abs() < 1e-3);
    assert!((closest[1] - (151.207)).abs() < 1e-3);
}
