extern crate ml_algo;

use ml_algo::matrix::DMatrix;
use ml_algo::linear::{LinearRegression, LinearRegressionOptions, Stepping};
use ml_algo::utils::{rmse_error, mae_error};

// This is taken from Machine Learning course by Anderw Ng
// https://www.coursera.org/learn/machine-learning
//
// Week 2, exercise 1

#[test]
fn food_truck_profit() {
    let mut lr = LinearRegression::new( LinearRegressionOptions::new()
                                              .max_iter(1500)
                                              .stepping(Stepping::Constant(0.01))
                                              .x_eps(1.0e-15)
                                              .eps(1.0e-15)
                                      );

    let train_x: DMatrix<f64> = DMatrix::from_csv("data/food_truck_profit.csv", 1, ',', Some(&[0])).unwrap();
    let train_y: DMatrix<f64> = DMatrix::from_csv("data/food_truck_profit.csv", 1, ',', Some(&[1])).unwrap();

    lr.fit(&train_x, train_y.data()).unwrap();
    let train_py = lr.predict(&train_x).unwrap();

    let rmse = rmse_error(train_y.data(), &train_py);
    let mae = mae_error(train_y.data(), &train_py);

    println!("Bias = {}, Coefficients = {:?}", lr.bias().unwrap(), lr.coefficients().unwrap());
    assert!((lr.bias().unwrap() - (-3.630291)).abs() < 1.0e-5);
    assert!((lr.coefficients().unwrap()[0] - 1.166362).abs() < 1.0e-5);
    println!("Train: RMSE = {}, MAE = {}", rmse, mae);

    let mut test_x: DMatrix<f64> = DMatrix::new_zeros(0, 1);
    test_x.append_row(&[3.5]);
    test_x.append_row(&[7.0]);
    let test_py = lr.predict(&test_x).unwrap();
    println!("For population = 35,000 we predict a profit of {}", test_py[0] * 10000.0);
    assert!((test_py[0] * 10000.0 - 4519.77).abs() < 0.1);
    println!("For population = 70,000 we predict a profit of {}", test_py[1] * 10000.0);
    assert!((test_py[1] * 10000.0 - 45342.45).abs() < 0.1);
}
