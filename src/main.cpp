//Copyright (c) 2015 Zachary Kann
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

// ---
// Author: Zachary Kann

#include "z_sim_params.hpp"
#include "z_file.hpp"
#include "z_vec.hpp"
#include "z_molecule.hpp"
#include "z_histogram.hpp"
#include "z_atom_group.hpp"
#include "z_gromacs.hpp"
#include "xdrfile_trr.h"
#include "boost/program_options.hpp"

namespace po = boost::program_options;

// Units are nm, ps.

int main (int argc, char *argv[]) {
  int st;
  SimParams params;
  int max_steps = std::numeric_limits<int>::max();

  std::string within_filename, output_filename;
  po::options_description desc("Options");
  desc.add_options()
    ("help,h",  "Print help messages")
    ("group,g", po::value<std::string>()->default_value("He"),
     "Group for temperature profiles")
    ("index,n", po::value<std::string>()->default_value("index.ndx"),
     ".ndx file containing atomic indices for groups")
    ("gro", po::value<std::string>()->default_value("conf.gro"),
     ".gro file containing list of atoms/molecules")
    ("top", po::value<std::string>()->default_value("topol.top"),
     ".top file containing atomic/molecular properties")
    ("within,w",
     po::value<std::string>(&within_filename)->default_value("within.dat"),
     ".dat file specifying solvating molecules")
    ("output,o",
     po::value<std::string>(&output_filename)->default_value("qDist.txt"),
     "Output filename");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(EXIT_SUCCESS);
  }

  std::map<std::string, std::vector<int> > groups;
  groups = ReadNdx(vm["index"].as<std::string>());

  std::vector<Molecule> molecules = GenMolecules(vm["top"].as<std::string>(),
                                                 params);
  AtomGroup all_atoms(vm["gro"].as<std::string>(), molecules);
  AtomGroup selected_group(vm["group"].as<std::string>(),
                           SelectGroup(groups, vm["group"].as<std::string>()),
                           all_atoms);

  std::fstream within_file(within_filename.c_str());
  assert(within_file.is_open());

  arma::irowvec within = arma::zeros<arma::irowvec>(selected_group.size());
  arma::irowvec nearest = arma::zeros<arma::irowvec>(4);
  Histogram q_hist(100, 0.01);

  rvec *x_in = NULL;
  matrix box_mat;
  arma::rowvec box = arma::zeros<arma::rowvec>(DIMS);
  std::string xtc_filename = "prod.xtc";
  XDRFILE *xtc_file;
  params.ExtractTrajMetadata(strdup(xtc_filename.c_str()), (&x_in), box);
  xtc_file = xdrfile_open(strdup(xtc_filename.c_str()), "r");
  params.set_max_time(vm["max_time"].as<double>());

  arma::rowvec dx;
  arma::mat dx_near = arma::zeros<arma::mat>(4,DIMS);
  arma::rowvec r2 = arma::zeros<arma::rowvec>(selected_group.size());
  float time,prec;
  for (int step=0; step<max_steps; step++) {
    if(read_xtc(xtc_file, params.num_atoms(), &st, &time, box_mat, x_in, &prec))
      break;
    params.set_box(box_mat);
    int i = 0;
    for (std::vector<int>::iterator i_atom = selected_group.begin();
         i_atom != selected_group.end(); i_atom++, i++) {
      selected_group.set_position(i, x_in[*i_atom]);
    }
    ReadInt(within, within_file);
    for (int i_atom = 0; i_atom < selected_group.size(); ++i_atom) {
      if (!within(i_atom)) continue;
      selected_group.FindNearestk(selected_group.position(i_atom), params.box(),
                                  selected_group.index_to_molecule(i_atom), 4,
                                  dx, r2, nearest);
      int q = 0;
      for (int i_near = 0; i_near < 4; ++i_near) {
        FindDxNoShift(dx, selected_group.position(i_atom),
                      selected_group.position(i_near), box);
        dx_near.row(i_near) = arma::normalise(dx);

        for (int i_near2 = 0; i_near2 < i_near; ++i_near2) {
          double sqrt_q =
              arma::dot(dx_near.row(i_near),dx_near.row(i_near2)) + 1.0/3.0;
          q += sqrt_q*sqrt_q;
        }
      }
      q_hist.Add(q);
    }
  }
  q_hist.Print(output_filename, true);
}
 // main
