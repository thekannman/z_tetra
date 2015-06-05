// Copyright (c) 2015 Zachary Kann
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// ---
// Author: Zachary Kann

// A collection of functions for basic i/o and file parsing.

#include <fstream>
#include <string>
#include <armadillo>

#ifndef _Z_FILE_HPP_
#define _Z_FILE_HPP_

// Skips over a set number of lines in an input file
extern void SkipLine (const std::string& filename, std::ifstream& file,
                      const int numSkip);

// Writes a matrix of ints to a file
extern void WriteInt (const arma::imat& int_mat, std::fstream& file,
                      const int dim1, const int dim2);

// Reads a matrix of ints from a file
extern void ReadInt (arma::imat& int_mat, std::fstream& file);

// Reads a vector of ints from a file
extern void ReadInt (arma::irowvec& int_vec, std::fstream& file);

// Grabs the final line from a file.
extern std::string ReadLastLine(std::string filename);

#endif
