/// A Rust module for speeding up SMILES processing
use pyo3::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::collections::{HashMap, HashSet};
use petgraph::graph::{Graph, UnGraph};
use petgraph::Undirected;
use petgraph::visit::EdgeRef;
use ndarray::Array2;
use nalgebra::DMatrix;
use numpy::{PyArray1, PyArray2, ToPyArray};


#[pyclass]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end: bool,
    token: Option<String>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            is_end: false,
            token: None,
        }
    }
}

#[pyclass]
struct SMILESTokenizer {
    vocab: HashMap<String, usize>,
    ids_to_tokens: HashMap<usize, String>,
    trie_root: TrieNode,
    #[pyo3(get)]
    model_max_length: usize,
}

const PAD_TOKEN: &str = "<|pad|>";
const BOS_TOKEN: &str = "<|bos|>";
const EOT_TOKEN: &str = "<|eot|>";
const SPLIT_TOKEN: &str = "<|split|>";
const NEW_TOKEN: &str = "<|new|>";
const MODEL_MAX_LENGTH: usize = 512;

#[pymethods]
impl SMILESTokenizer {
    #[new]
    fn new(vocab_path: &str) -> PyResult<Self> {
        let mut tokenizer = SMILESTokenizer {
            vocab: HashMap::new(),
            ids_to_tokens: HashMap::new(),
            trie_root: TrieNode::new(),
            model_max_length: MODEL_MAX_LENGTH,
        };

        // Initialize special tokens
        let special_tokens = vec![PAD_TOKEN, BOS_TOKEN, SPLIT_TOKEN, NEW_TOKEN, EOT_TOKEN];
        for (i, token) in special_tokens.iter().enumerate() {
            tokenizer.vocab.insert(token.to_string(), i);
            tokenizer.ids_to_tokens.insert(i, token.to_string());
        }

        // Load vocabulary from file
        let file = File::open(vocab_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let reader = io::BufReader::new(file);
        let special_tokens_len = special_tokens.len();

        for (i, line) in reader.lines().enumerate() {
            let token = line.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            let id = i + special_tokens_len;
            tokenizer.vocab.insert(token.clone(), id);
            tokenizer.ids_to_tokens.insert(id, token);
        }

        tokenizer.build_trie();
        Ok(tokenizer)
    }

    fn build_trie(&mut self) {
        for token in self.vocab.keys() {
            let mut current = &mut self.trie_root;
            
            for c in token.chars() {
                current = current.children.entry(c)
                    .or_insert_with(TrieNode::new);
            }
            
            current.is_end = true;
            current.token = Some(token.clone());
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();

        while i < chars.len() {
            let mut current = &self.trie_root;
            let mut longest_match = None;
            let mut longest_end = i;
            let mut j = i;

            while j < chars.len() {
                if let Some(next) = current.children.get(&chars[j]) {
                    current = next;
                    if current.is_end {
                        longest_match = current.token.clone();
                        longest_end = j + 1;
                    }
                    j += 1;
                } else {
                    break;
                }
            }

            if let Some(token) = longest_match {
                tokens.push(token);
                i = longest_end;
            } else {
                tokens.push(chars[i].to_string());
                i += 1;
            }
        }

        tokens
    }

    #[pyo3(name = "encode")]
    fn encode_py(&self, text: &str, py: Python) -> PyResult<PyObject> {
        let tokens = self.tokenize(text)
            .into_iter()
            .map(|token| self.convert_token_to_id(&token))
            .collect::<Vec<usize>>();
        
        Ok(tokens.into_py(py))
    }

    #[pyo3(name = "decode")]
    fn decode_py(&self, ids: Vec<usize>) -> String {
        ids.iter()
            .map(|&id| self.convert_id_to_token(id))
            .collect::<Vec<String>>()
            .join("")
    }

    fn convert_token_to_id(&self, token: &str) -> usize {
        *self.vocab.get(token).unwrap_or(&self.vocab[PAD_TOKEN])
    }

    fn convert_id_to_token(&self, id: usize) -> String {
        self.ids_to_tokens.get(&id)
            .unwrap_or(&PAD_TOKEN.to_string())
            .clone()
    }

    #[getter]
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    #[getter]
    fn get_vocab(&self, py: Python) -> PyResult<PyObject> {
        let dict = self.vocab.clone().into_py(py);
        Ok(dict)
    }

    #[getter]
    fn pad_token(&self) -> &'static str {
        PAD_TOKEN
    }

    #[getter]
    fn bos_token(&self) -> &'static str {
        BOS_TOKEN
    }

    #[getter]
    fn eos_token(&self) -> &'static str {
        EOT_TOKEN
    }
}

/*
#[derive(Debug)]
struct Atom {
    index: usize,
    atom_idx: usize,
    is_aromatic: bool,
}
*/

#[pyclass]
struct SMILESParser {
    atom_dict: HashMap<String, usize>,
}

#[pymethods]
impl SMILESParser {
    #[new]
    fn new(vocab_path: &str) -> PyResult<Self> {
        let file = File::open(vocab_path)?;
        let reader = io::BufReader::new(file);
        let mut atom_dict = HashMap::new();
        
        // Ensure <|unk|> is at index 0
        atom_dict.insert("<|unk|>".to_string(), 0);
        
        for (index, line) in reader.lines().enumerate() {
            let symbol = line?;
            if symbol != "<|pad|>" && symbol != "<|unv|>" && symbol != "<|unk|>" {
                atom_dict.insert(symbol, index + 1);
            }
        }
        
        Ok(Self { atom_dict })
    }

    fn parse<'py>(&self, py: Python<'py>, smiles: &str) -> PyResult<(&'py PyArray1<usize>, &'py PyArray2<f64>)> {
        let (indices, distances) = self.parse_internal(smiles)?;
        Ok((
            indices.to_pyarray(py),
            distances.to_pyarray(py)
        ))
    }

    #[getter]
    fn get_vocab_size(&self) -> usize {
        self.atom_dict.len() + 3
    }
}

impl SMILESParser {
    fn parse_internal(&self, smiles: &str) -> Result<(Vec<usize>, Array2<f64>), PyErr> {
        let (atoms, graph) = self.build_molecular_graph(smiles)?;
        let distances = self.calculate_resistance_distances(&graph, atoms.len() - 1);  // -1 to account for universal token
        Ok((atoms, distances))
    }

    fn calculate_resistance_distances(&self, graph: &UnGraph<(), f64>, num_atoms: usize) -> Array2<f64> {
        // Create Laplacian matrix
        let mut laplacian = DMatrix::<f64>::zeros(num_atoms, num_atoms);
        
        // Fill Laplacian matrix
        for edge in graph.edge_references() {
            let (i, j) = (edge.source().index(), edge.target().index());
            let weight = *edge.weight(); // bond order
            let conductance = weight; // conductance = bond order
            
            laplacian[(i, i)] += conductance;
            laplacian[(j, j)] += conductance;
            laplacian[(i, j)] -= conductance;
            laplacian[(j, i)] -= conductance;
        }
        
        // Calculate pseudoinverse using eigendecomposition
        let eigen = laplacian.symmetric_eigen();
        let mut pinv = DMatrix::zeros(num_atoms, num_atoms);
        
        for i in 0..num_atoms {
            if eigen.eigenvalues[i] > 1e-10 {  // numerical threshold for "non-zero"
                let v = eigen.eigenvectors.column(i);
                pinv += (1.0 / eigen.eigenvalues[i]) * (v * v.transpose());
            }
        }
        
        // Calculate resistance distances
        let mut distances = Array2::zeros((num_atoms + 1, num_atoms + 1));
        for i in 0..num_atoms {
            for j in i+1..num_atoms {
                let dist = pinv[(i,i)] + pinv[(j,j)] - 2.0 * pinv[(i,j)];
                distances[[i+1, j+1]] = dist;
                distances[[j+1, i+1]] = dist;
            }
        }
        
        // Add universal token distances (all zeros)
        for i in 0..=num_atoms {
            distances[[0, i]] = 0.0;
            distances[[i, 0]] = 0.0;
        }
        
        distances
    }

    fn get_atom_index(&self, symbol: &str) -> usize {
        *self.atom_dict.get(symbol.to_lowercase().as_str()).unwrap_or(self.atom_dict.get("<|unk|>").unwrap())
    }

    fn build_molecular_graph(&self, smiles: &str) -> Result<(Vec<usize>, UnGraph<(), f64>), PyErr> {
        let mut atoms: Vec<usize> = vec![1];  // Start with universal token
        let mut graph = Graph::<(), f64, Undirected>::new_undirected();
        let mut branch_points: Vec<usize> = Vec::new();
        let mut ring_closures: HashMap<usize, (usize, Option<char>, bool)> = HashMap::new();
        let mut current_atom_idx: usize = 0;
        let mut aromatic_system: HashSet<usize> = HashSet::new();
        let mut prev_atom_idx: Option<usize> = None;
        let mut next_bond_type: Option<char> = None;
        
        let mut chars = smiles.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                'A'..='Z' | 'a'..='z' => {
                    let mut symbol = c.to_string();

                    // Check for two-letter atoms (Cl, Br, etc.)
                    if let Some(&next_c) = chars.peek() {
                        // Only combine if the current symbol + next_c exists in our vocabulary
                        let potential_symbol = format!("{}{}", symbol, next_c).to_lowercase();
                        if next_c.is_ascii_lowercase() && self.atom_dict.contains_key(&potential_symbol) {
                            symbol.push(chars.next().unwrap());
                        }
                    }
                    
                    let is_aromatic = symbol.chars().next().unwrap().is_ascii_lowercase();
                    let symbol = if is_aromatic {
                        symbol.to_ascii_uppercase()
                    } else {
                        symbol
                    };
                    
                    // Add node to graph
                    graph.add_node(());
                    current_atom_idx = graph.node_count() - 1;
                    
                    // Add atom to list
                    atoms.push(self.get_atom_index(&symbol));
                    
                    // Handle bond from previous atom if it exists
                    if let Some(prev_idx) = prev_atom_idx {
                        let bond_order = match next_bond_type.take() {
                            Some('=') => 2.0,
                            Some('#') => 3.0,
                            _ if is_aromatic && aromatic_system.contains(&prev_idx) => 1.5,
                            _ => 1.0,
                        };
                        graph.add_edge((prev_idx as u32).into(), (current_atom_idx as u32).into(), bond_order);
                    }
                    
                    if is_aromatic {
                        aromatic_system.insert(current_atom_idx);
                    }
                    
                    prev_atom_idx = Some(current_atom_idx);
                },
                '=' => next_bond_type = Some('='),
                '#' => next_bond_type = Some('#'),
                '(' => {
                    branch_points.push(current_atom_idx);
                },
                ')' => {
                    if let Some(branch_point) = branch_points.pop() {
                        current_atom_idx = branch_point;
                        prev_atom_idx = Some(current_atom_idx);
                    }
                },
                '[' => {
                    let mut bracket_content = String::new();
                    let mut bracket_depth = 1;
                    
                    while let Some(c) = chars.next() {
                        match c {
                            '[' => bracket_depth += 1,
                            ']' => {
                                bracket_depth -= 1;
                                if bracket_depth == 0 { break; }
                            },
                            _ => bracket_content.push(c),
                        }
                    }
                    
                    if bracket_depth > 0 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unclosed bracket in SMILES"));
                    }
                    
                    // Parse bracket content
                    let mut symbol = String::new();
                    let mut chars_iter = bracket_content.chars().peekable();
                    
                    // Handle charge and isotope indicators if present
                    while let Some(c) = chars_iter.next() {
                        match c {
                            'A'..='Z' | 'a'..='z' => {
                                symbol.push(c);
                                if let Some(&next_c) = chars_iter.peek() {
                                    if next_c.is_ascii_lowercase() {
                                        let potential_symbol = format!("{}{}", symbol, next_c).to_lowercase();
                                        if self.atom_dict.contains_key(&potential_symbol) {
                                            symbol.push(chars_iter.next().unwrap());
                                        }
                                    }
                                }
                                break;
                            },
                            _ => continue, // Skip isotope numbers for now
                        }
                    }
                    
                    // Add node to graph
                    graph.add_node(());
                    current_atom_idx = graph.node_count() - 1;
                    
                    // Add atom to list
                    atoms.push(self.get_atom_index(&symbol));
                    
                    // Handle bond from previous atom if it exists
                    if let Some(prev_idx) = prev_atom_idx {
                        let bond_order = match next_bond_type.take() {
                            Some('=') => 2.0,
                            Some('#') => 3.0,
                            _ => 1.0,
                        };
                        graph.add_edge((prev_idx as u32).into(), (current_atom_idx as u32).into(), bond_order);
                    }
                    
                    prev_atom_idx = Some(current_atom_idx);
                },
                '%' => {
                    // Handle ring closures >= 10
                    let digit1 = chars.next()
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected first digit after %"))?
                        .to_digit(10)
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid first digit after %"))?;
                    let digit2 = chars.next()
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Expected second digit after %"))?
                        .to_digit(10)
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid second digit after %"))?;
                    let ring_number = (digit1 * 10 + digit2) as usize;
                    self.handle_ring_closure(&mut ring_closures, &mut graph, ring_number, 
                                          current_atom_idx, &aromatic_system, next_bond_type.take())?;
                },
                '0'..='9' => {
                    let ring_number = c.to_digit(10).unwrap() as usize;
                    self.handle_ring_closure(&mut ring_closures, &mut graph, ring_number,
                                          current_atom_idx, &aromatic_system, next_bond_type.take())?;
                },
                '.' => {
                    // Reset previous atom index since next atom starts a new component
                    prev_atom_idx = None;
                    // Clear any pending bond type
                    next_bond_type = None;
                },
                '-' => next_bond_type = Some('-'),  // Explicit single bond
                '/' | '\\' => {
                    // For now, treat stereochemistry indicators as single bonds
                    // TODO: Add proper stereochemistry handling if needed
                    next_bond_type = Some('-');
                },
                _ if c.is_whitespace() => continue,
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported character in SMILES: {}", c)
                )),
            }
        }
        
        // Check for unclosed rings
        if !ring_closures.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unclosed ring in SMILES"));
        }
        
        Ok((atoms, graph))
    }

    fn handle_ring_closure(
        &self,
        ring_closures: &mut HashMap<usize, (usize, Option<char>, bool)>,
        graph: &mut UnGraph<(), f64>,
        ring_number: usize,
        current_atom_idx: usize,
        aromatic_system: &HashSet<usize>,
        bond_type: Option<char>,
    ) -> Result<(), PyErr> {
        if let Some((other_atom_idx, other_bond_type, _is_aromatic_start)) = ring_closures.remove(&ring_number) {
            // Close the ring
            let bond_order = match bond_type.or(other_bond_type) {
                Some('-') => 1.0,
                Some('=') => 2.0,
                Some('#') => 3.0,
                _ if aromatic_system.contains(&current_atom_idx) && 
                     aromatic_system.contains(&other_atom_idx) => 1.5,  // Changed this condition
                _ => 1.0,
            };
            graph.add_edge(
                (other_atom_idx as u32).into(),
                (current_atom_idx as u32).into(),
                bond_order
            );
        } else {
            // Start a new ring
            ring_closures.insert(
                ring_number,
                (current_atom_idx, bond_type, aromatic_system.contains(&current_atom_idx))
            );
        }
        Ok(())
    }
}


/// A Rust module for interfacing with SMILESTokenizer and SMILESParser
#[pymodule]
fn smiles_decoder_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SMILESTokenizer>()?;
    m.add_class::<SMILESParser>()?;
    Ok(())
}