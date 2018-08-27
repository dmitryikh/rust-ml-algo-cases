extern crate ml_algo;

use ml_algo::matrix::DMatrix;
use ml_algo::naive_bayes::{NaiveBayes, Gaussian};
use ml_algo::utils::accuracy;

#[test]
fn naive_bayes_digits() {
    // Data preparation [python]:
    //    from sklearn.datasets import load_digits
    //    from sklearn.model_selection import train_test_split
    //    import pandas as pd
    //    import numpy as np
    //    digits = load_digits()
    //    X, y = digits.data, digits.target
    //    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    //    train = pd.DataFrame(data={'labels': y_train}, dtype=np.int)
    //    for i in range(X_train.shape[1]):
    //        train[f'pixel{i}'] = X_train[:, i]
    //    test = pd.DataFrame(data={'labels': y_test}, dtype=np.int)
    //    for i in range(X_test.shape[1]):
    //        test[f'pixel{i}'] = X_test[:, i]
    //    train.to_csv('./data/digits_train.csv', index=None)
    //    test.to_csv('./data/digits_test.csv', index=None)

    // Predictions [python]:
    //    from sklearn import naive_bayes
    //    from sklearn.metrics import accuracy_score
    //    b = naive_bayes.GaussianNB()
    //    b.fit(X=X_train, y=y_train)
    //    y_p = b.predict(X_test)
    //    print(accuracy_score(y_test, y_p))
    //  Out: 0.825


    let mut bayes = NaiveBayes::<Gaussian>::new();
    let train_x: DMatrix<f64> = DMatrix::from_csv("data/digits_train.csv", 1, ',', Some(&(1..64).collect::<Vec<usize>>())).unwrap();
    let train_y_: DMatrix<f64> = DMatrix::from_csv("data/digits_train.csv", 1, ',', Some(&[0])).unwrap();
    let train_y: Vec<_> = train_y_.data().iter().map(|&v| v as u32).collect();
    let test_x: DMatrix<f64> = DMatrix::from_csv("data/digits_test.csv", 1, ',', Some(&(1..64).collect::<Vec<usize>>())).unwrap();
    let test_y_: DMatrix<f64> = DMatrix::from_csv("data/digits_test.csv", 1, ',', Some(&[0])).unwrap();
    let test_y: Vec<_> = test_y_.data().iter().map(|&v| v as u32).collect();

    bayes.fit(&train_x, &train_y).unwrap();
    let test_py = bayes.predict(&test_x).unwrap();

    // for i in 0..test_py.len() {
    //     println!("{} -> {}", test_y[i], test_py[i]);
    // }
    let acc = accuracy(&test_y, &test_py);
    println!("Accuracy for digits prediction: {}", acc);
    assert!(acc > 0.825);
}
