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

#include "z_file.hpp"
#include <cassert>
#include <string>
#include "z_string.hpp"

void SkipLine (const std::string& filename, std::ifstream& file,
               const int numSkip) {
  std::string line;
  assert(file.is_open());
  for (int i1=0; i1<numSkip; i1++) {
    while(true) {
      getline(file, line);
      if(!(line[0] == '#' || line[0] == '@')) break;
    }
  }
}

void WriteInt (const arma::imat& int_mat, std::fstream& file, const int dim1,
               const int dim2 = 1) {
  assert(file.is_open());
  for (int i1=0; i1<dim1; i1++) {
    for (int i2=0; i2<dim2; i2++)
      file << int_mat(i1,i2) << " ";
  }
  file << std::endl;
}

void ReadInt (arma::imat& int_mat, std::fstream& file) {
  int dim1 = int_mat.n_cols;
  int dim2 = int_mat.n_rows;
  std::string line;
  assert(file.is_open());
  getline(file, line);
  std::vector<std::string> split_line = Split(line, ' ');
  std::vector<std::string>::iterator i_split = split_line.begin();
  for (int i1=0; i1<dim1; i1++) {
    for (int i2=0; i2<dim2; i2++) {
      int_mat(i1,i2) = atoi((*i_split).c_str());
      i_split++;
    }
  }
}

void ReadInt (arma::irowvec& int_vec, std::fstream& file) {
  int dim = int_vec.n_elem;
  std::string line;
  assert(file.is_open());
  getline(file, line);
  std::vector<std::string> split_line = Split(line, ' ');
  std::vector<std::string>::iterator i_split = split_line.begin();
  for (int i1=0; i1<dim; i1++) {
    int_vec(i1) = atoi((*i_split).c_str());
    i_split++;
  }
}

std::string ReadLastLine(std::string filename) {
  std::string line;
  std::string last_line;
  std::ifstream file(filename.c_str());
  assert(file.is_open());
  while(getline(file, line)) {
    bool is_empty = true;
    for (unsigned i = 0; i < line.size(); i++) {
      char ch = line[i];
      is_empty = is_empty && isspace(ch);
    }
    if (!is_empty)
      last_line = line;
  }
  file.close();
  return last_line;
}
