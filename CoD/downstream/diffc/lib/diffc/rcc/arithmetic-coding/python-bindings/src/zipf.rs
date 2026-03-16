//#![feature(exclusive_range_pattern)]

use std::ops::Range;

use arithmetic_coding::Model;

//mod common;

#[derive(Clone)]
pub struct ZipfModel {
    s_values: Vec<f64>,
    n_values: Vec<u32>,
    current_index: usize,
    cmf: Vec<u64>,
    exhausted: bool,
}


impl ZipfModel {
    pub fn new(s_values: Vec<f64>, n_values: Vec<u32>) -> Self {
        assert_eq!(s_values.len(), n_values.len(), "s_values and n_values must have the same length");
        assert!(!s_values.is_empty(), "At least one (s, n) pair is required");

        let mut model = ZipfModel {
            s_values,
            n_values,
            current_index: 0,
            cmf: Vec::new(),
            exhausted: false,
        };
        model.update_cmf();
        model
    }

    fn update_cmf(&mut self) {
        let s = self.s_values[self.current_index];
        let n = self.n_values[self.current_index];
        
        let max_denom = Self::max_denominator();
        self.cmf = vec![0; n as usize + 1];
        
        let mut sum = 0.0;
        let harmonic_n: f64 = (1..=n).map(|k| 1.0 / (k as f64).powf(s)).sum();

        for k in 1..=n {
            sum += 1.0 / (k as f64).powf(s) / harmonic_n;
            self.cmf[k as usize] = (sum * max_denom as f64).min(max_denom as f64) as u64;
        }

        // Ensure the last value is exactly max_denominator
        *self.cmf.last_mut().unwrap() = max_denom;
    }

    fn max_denominator() -> u64 {
        1 << 24
    }

    fn is_significantly_different(a: f64, b: f64) -> bool {
        (a - b).abs() > 1e-10
    }
    
    fn should_update_cmf(&self) -> bool {
        let current_s = self.s_values[self.current_index];
        let previous_s = self.s_values[self.current_index - 1];
        let current_n = self.n_values[self.current_index];
        let previous_n = self.n_values[self.current_index - 1];
    
        Self::is_significantly_different(current_s, previous_s) || current_n != previous_n
    }
}

#[derive(Debug, thiserror::Error)]
#[error("invalid symbol: {0}")]
pub struct Error(pub u32);

impl Model for ZipfModel {
    type Symbol = u32;
    type ValueError = Error;
    type B = u64;

    fn probability(&self, symbol: Option<&Self::Symbol>) -> Result<Range<Self::B>, Error> {
        if self.exhausted {
            match symbol {
                None => Ok(0..Self::max_denominator()),
                Some(&s) => Err(Error(s)),
            }
        } else {
            match symbol {
                None => Err(Error(self.n_values[self.current_index])), // Error for None when not exhausted
                Some(&s) if s < self.n_values[self.current_index] => {
                    Ok(self.cmf[s as usize]..self.cmf[s as usize + 1])
                },
                Some(&s) => Err(Error(s)),
            }
        }
    }

    fn symbol(&self, value: Self::B) -> Option<Self::Symbol> {
        if self.exhausted {
            return None;
        }
        
        if value >= Self::max_denominator() {
            return None;
        }
        
        match self.cmf.binary_search(&value) {
            Ok(exact_index) => Some(exact_index as u32), // minus 1?
            Err(insertion_index) => Some(insertion_index as u32 - 1),
        }
    }

    fn max_denominator(&self) -> Self::B {
        Self::max_denominator()
    }

    fn update(&mut self, symbol: Option<&Self::Symbol>) {
        if self.exhausted {
            return;
        }

        if symbol.is_none() {
            self.exhausted = true;
            return;
        }

        self.current_index += 1;
        if self.current_index >= self.s_values.len() {
            self.exhausted = true;
        } else if self.should_update_cmf() {
            self.update_cmf();
        }
    }
}


// fn main() {
//     let s_values = vec![1.0, 1.1, 1.2, 1.0, 1.0, 1.1, 1.2, 1.0, 1.1, 1.2, 1.0, 1.0, 1.1, 1.2];
//     let n_values = vec![3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3];
//     let model = ZipfModel::new(s_values, n_values);

//     common::round_trip(model, vec![2, 1, 1, 2, 2, 0, 1, 2, 1, 1, 2, 2, 0, 1]);
// }
