use pyo3::prelude::*;
use arithmetic_coding::{Encoder, Decoder};
use bitstream_io::{BigEndian, BitReader, BitWrite, BitWriter};

mod zipf;
use zipf::ZipfModel;

#[pyfunction]
fn encode_zipf(s_values: Vec<f64>, n_values: Vec<u32>, numbers: Vec<u32>) -> PyResult<Vec<u8>> {
    let model = ZipfModel::new(s_values, n_values);
    let mut bitwriter = BitWriter::endian(Vec::new(), BigEndian);
    let mut encoder = Encoder::new(model, &mut bitwriter);

    encoder.encode_all(numbers).map_err(|_e| PyErr::new::<pyo3::exceptions::PyValueError, _>("Encoding error"))?;
    bitwriter.byte_align().map_err(|_e| PyErr::new::<pyo3::exceptions::PyValueError, _>("Byte alignment error"))?;

    Ok(bitwriter.into_writer())
}

#[pyfunction]
fn decode_zipf(s_values: Vec<f64>, n_values: Vec<u32>, encoded: Vec<u8>) -> PyResult<Vec<u32>> {
    let model = ZipfModel::new(s_values, n_values);
    let bitreader = BitReader::endian(encoded.as_slice(), BigEndian);
    let mut decoder = Decoder::new(model, bitreader);
    
    decoder.decode_all()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_e| PyErr::new::<pyo3::exceptions::PyValueError, _>("Decoding error"))
}

#[pymodule]
fn zipf_encoding(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_zipf, m)?)?;
    m.add_function(wrap_pyfunction!(decode_zipf, m)?)?;
    Ok(())
}