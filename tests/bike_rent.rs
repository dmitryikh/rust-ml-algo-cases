extern crate ml_algo;

use ml_algo::matrix::DMatrix;
use ml_algo::rforest::{RegRandomForest, RandomForestOptions};
use ml_algo::tree::{CartOptions, SplitCriteria};
use ml_algo::utils::{rmse_error, mae_error};

#[test]
fn bike_rent() {
    let mut forest = RegRandomForest::new( RandomForestOptions::new()
                                              .n_trees(50)
                                              .tree_options( CartOptions::new()
                                                                 .max_depth(20)
                                                                 .min_in_leaf(1)
                                                                 .random_seed(42)
                                                                 .split_criterion(SplitCriteria::MSE)
                                                                 // .split_features(SplitFeatures::Random(5))
                                                           )
                                         );
    let train_x: DMatrix<f64> = DMatrix::from_csv("data/bike_rent_train.csv", 1, ',', Some(&[0, 1, 2, 3, 4, 5, 6])).unwrap();
    let train_y: DMatrix<f64> = DMatrix::from_csv("data/bike_rent_train.csv", 1, ',', Some(&[7])).unwrap();
    let test_x: DMatrix<f64> = DMatrix::from_csv("data/bike_rent_test.csv", 1, ',', Some(&[0, 1, 2, 3, 4, 5, 6])).unwrap();
    let test_y: DMatrix<f64> = DMatrix::from_csv("data/bike_rent_test.csv", 1, ',', Some(&[7])).unwrap();
    forest.fit(&train_x, train_y.data()).unwrap();
    let train_py = forest.predict(&train_x).unwrap();
    // write_csv_col("output/forest_bike_rent_train.csv", &train_py, None).unwrap();
    let rmse = rmse_error(train_y.data(), &train_py);
    let mae = mae_error(train_y.data(), &train_py);
    println!("Train: RMSE = {}, MAE = {}", rmse, mae);

    let test_py = forest.predict(&test_x).unwrap();
    // write_csv_col("output/forest_bike_rent_test.csv", &test_py, None).unwrap();
    let rmse = rmse_error(test_y.data(), &test_py);
    let mae = mae_error(test_y.data(), &test_py);
    println!("Test: RMSE = {}, MAE = {}", rmse, mae);
}
