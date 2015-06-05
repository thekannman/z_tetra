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

// Implementation of AtomGroup class. See include/z_atom_group.hpp for
// more details about the class.

#include "z_atom_group.hpp"
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include "z_string.hpp"
#include "z_vec.hpp"
#include <algorithm>

void AtomGroup::Init(std::string name, std::vector<double> index_to_mass,
                     std::vector<int> index_to_molecule,
                     std::vector<double> index_to_sigma,
                     std::vector<double> index_to_epsilon,
                     std::vector<double> index_to_charge,
                     std::vector<std::string> atom_names,
                     std::vector<std::string> molecule_names) {
  for (std::vector<int>::iterator i_index = indices_.begin();
       i_index != indices_.end(); ++i_index) {
    index_to_mass_.push_back(index_to_mass[*i_index]);
    index_to_molecule_.push_back(index_to_molecule[*i_index]);
    index_to_sigma_.push_back(index_to_sigma[*i_index]);
    index_to_epsilon_.push_back(index_to_epsilon[*i_index]);
    index_to_charge_.push_back(index_to_charge[*i_index]);
    atom_names_.push_back(atom_names[*i_index]);
  }
  std::vector<double>::iterator i_mass = index_to_mass_.begin();
  for (std::vector<int>::iterator i_mol = index_to_molecule_.begin();
       i_mol != index_to_molecule_.end(); ++i_mol, ++i_mass) {
    std::vector<int>::iterator mol_index;
    mol_index = std::find(molecular_index_to_molecule_.begin(),
                          molecular_index_to_molecule_.end(), *i_mol);
  if (mol_index == molecular_index_to_molecule_.end()) {
    molecular_index_to_mass_.push_back(0.0);
    index_to_molecular_index_.push_back(molecular_index_to_molecule_.size());
    molecular_index_to_molecule_.push_back(*i_mol);
    molecule_names_.push_back(molecule_names[*i_mol]);
  } else
    index_to_molecular_index_.push_back(
        std::distance(molecular_index_to_molecule_.begin(),mol_index));
    molecular_index_to_mass_.back() += *i_mass;
  }
  num_molecules_ = molecular_index_to_molecule_.size();
  com_positions_.set_size(num_molecules_, DIMS);
  com_velocities_.set_size(num_molecules_, DIMS);
}

void AtomGroup::RemoveMolecule(const int molecule_id,
                               arma::rowvec& old_com_position,
                               arma::rowvec& old_com_velocity) {
  std::vector<int> mark_for_rem;
  const int marked_molecule = molecular_index_to_molecule_[molecule_id];
  for (std::vector<int>::iterator i_mol = index_to_molecule_.begin();
       i_mol != index_to_molecule_.end(); i_mol++) {
    if (*i_mol == marked_molecule)
      mark_for_rem.push_back(std::distance(index_to_molecule_.begin(),i_mol));
    else if (*i_mol > marked_molecule)
      (*i_mol) -= 1;
  }
  for (std::vector<int>::iterator i_mol = index_to_molecular_index_.begin(),
           i_atom = indices_.begin();
       i_mol != index_to_molecular_index_.end(); ++i_mol, ++i_atom) {
    // Decrements molecular and atomic indices to eliminate gap
    if ((*i_mol) > molecule_id) {
      (*i_mol) -= 1;
      (*i_atom) -= mark_for_rem.size();
    }
  }
  assert(!mark_for_rem.empty());
  for (std::vector<int>::iterator i_mol =
           molecular_index_to_molecule_.begin()+molecule_id+1;
       i_mol != molecular_index_to_molecule_.end(); ++i_mol) {
    (*i_mol) -= 1;
  }
  molecular_index_to_mass_.erase(
      molecular_index_to_mass_.begin() + molecule_id);
  molecular_index_to_molecule_.erase(
      molecular_index_to_molecule_.begin() + molecule_id);
  molecule_names_.erase(
      molecule_names_.begin() + molecule_id);
  // Save references to position/velocity for later replacement.
  old_com_position = com_positions_.row(molecule_id);
  old_com_velocity = com_velocities_.row(molecule_id);
  com_positions_.shed_row(molecule_id);
  com_velocities_.shed_row(molecule_id);
  for (std::vector<int>::reverse_iterator i_atom = mark_for_rem.rbegin();
       i_atom != mark_for_rem.rend(); i_atom++) {
    RemoveAtom(*i_atom);
  }
  num_molecules_--;
}


void AtomGroup::AddMolecule(const int molecule_id, const Molecule& molecule,
                            const arma::rowvec& position,
                            const arma::rowvec& velocity)
{
    // TODO(Zak): Allow for polyatomic molecules by randomly
    // orienting around center of mass position.
    // This assert is a placeholder until then.
    assert (molecule.num_atoms() == 1);
    molecule_names_.push_back(molecule.name());
    molecular_index_to_mass_.push_back(molecule.mass());
    molecular_index_to_molecule_.push_back(molecule_id);
    molecule_names_.push_back(molecule.name());
    for (std::vector<Atom>::const_iterator i_atom = molecule.begin();
         i_atom != molecule.end(); ++i_atom) {
      AddAtom(*i_atom, molecule_id, position, velocity);
    }
    num_molecules_++;
    com_positions_.insert_rows(com_positions_.n_rows, position);
    com_velocities_.insert_rows(com_velocities_.n_rows, velocity);
}

void AtomGroup::WriteGro(const std::string& gro, const arma::rowvec& box,
                         const std::string description) {
    std::ofstream gro_file;
    gro_file.open(gro.c_str());
    assert(gro_file.is_open());
    gro_file << description << std::endl;
    gro_file << group_size_ << std::endl;
    for (int i_atom = 0; i_atom < group_size_; i_atom++) {
      gro_file << boost::format("%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f") %
          (index_to_molecular_index_[i_atom]+1) %
          molecule_names_[index_to_molecular_index_[i_atom]] %
          atom_names_[i_atom] % (indices_[i_atom]+1) % positions_(i_atom,0) %
          positions_(i_atom,1) % positions_(i_atom,2) %
          velocities_(i_atom,0) % velocities_(i_atom,1) %
          velocities_(i_atom,2) << std::endl;
    }
    gro_file << boost::format("%10.5f%10.5f%10.5f") % box(0) % box(1) % box(2);
    gro_file.close();
}

void AtomGroup::UpdateCom(const bool skip_reweighting) {
  ZeroCom();
  // Actually com_position*mass and com_velocity*mass
  for (int i_atom = 0; i_atom < group_size_; i_atom++) {
    com_positions_.row(index_to_molecular_index_[i_atom]) +=
        index_to_mass_[i_atom]*positions_.row(i_atom);
    com_velocities_.row(index_to_molecular_index_[i_atom]) +=
        index_to_mass_[i_atom]*velocities_.row(i_atom);
  }
  if (!skip_reweighting) {
    for (int i_mol = 0; i_mol < num_molecules_; ++i_mol) {
      com_positions_.row(i_mol) /= molecular_index_to_mass_[i_mol];
      com_velocities_.row(i_mol) /= molecular_index_to_mass_[i_mol];
    }
  }
}

void AtomGroup::CalculateElectricField(const AtomGroup& other_group,
                              const arma::rowvec& box,
                              const arma::imat& nearby_molecules,
                              arma::rowvec& dx,
                              const bool zero_first) {
  if (zero_first)
    ZeroElectricField();
  for (int i_atom = 0; i_atom < group_size_; ++i_atom) {
    for (int i_other = 0; i_other < other_group.size(); ++i_other) {
      if (nearby_molecules(i_atom, other_group.index_to_molecule(i_other)) == 0)
        continue;
      //Skip neutral atoms
      if (std::abs(other_group.charge(i_other) < 0.0001)) continue;
      FindDxNoShift(dx, position(i_atom), other_group.position(i_other), box);
      double r2 = arma::dot(dx,dx);
      electric_fields_.row(i_atom) +=
          other_group.charge(i_other)*dx/r2/std::sqrt(r2);
    }
  }
}

void AtomGroup::CalculateElectricField(const AtomGroup& other_group,
                              const arma::rowvec& box,
                              const double cutoff_squared,
                              arma::rowvec& dx,
                              const bool zero_first) {
  if (zero_first)
    ZeroElectricField();
  for (int i_atom = 0; i_atom < group_size_; ++i_atom) {
    for (int i_other = 0; i_other < other_group.size(); ++i_other) {
      //Skip neutral atoms
      if (std::abs(other_group.charge(i_other) < 0.0001)) continue;
      FindDxNoShift(dx, position(i_atom), other_group.position(i_other), box);
      double r2 = arma::dot(dx,dx);
      if (r2 > cutoff_squared) continue;
      electric_fields_.row(i_atom) +=
          other_group.charge(i_other)*dx/r2/std::sqrt(r2);
    }
  }
}

void AtomGroup::MarkNearbyAtoms(const AtomGroup& other_group,
                                const arma::rowvec& box,
                                const double cutoff_squared,
                                arma::imat& nearby_atoms) const {
  nearby_atoms = arma::zeros<arma::imat>(group_size_, other_group.size());
  arma::rowvec dx;
  for (int i_atom = 0; i_atom < group_size_; ++i_atom) {
    const int molecule_number = index_to_molecule_[i_atom];
    for (int i_other = 0; i_other < other_group.size(); ++i_other) {
      if (molecule_number == other_group.index_to_molecule(i_other)) continue;
      FindDxNoShift(dx, position(i_atom), other_group.position(i_other), box);
      double r2 = arma::dot(dx,dx);
      if (r2 < cutoff_squared)
        nearby_atoms(i_atom,i_other) = 1;
    }
  }
}

void AtomGroup::ReadGro(const std::string& gro,
                        const std::vector<Molecule> molecules) {
  std::ifstream gro_file;
  std::string line;
  gro_file.open(gro.c_str());
  assert(gro_file.is_open());
  getline(gro_file, line);
  getline(gro_file, line);
  std::vector<double> index_to_mass;
  std::vector<int> index_to_molecule;
  std::vector<double> index_to_sigma;
  std::vector<double> index_to_epsilon;
  std::vector<double> index_to_charge;
  std::vector<std::string> atom_names;
  std::vector<std::string> molecule_names;
  const int num_atoms = boost::lexical_cast<int>(Trim(line));
  Molecule mol_match;
  positions_.zeros(num_atoms,DIMS);
  velocities_.zeros(num_atoms,DIMS);
  int mol_number = std::numeric_limits<int>::max(), mol_number_old;
  for (int i_atom = 0; i_atom < num_atoms; i_atom++) {
    indices_.push_back(i_atom);
    std::string mol_name;
    assert(!gro_file.eof());
    getline(gro_file, line);
    std::vector<std::string> split_line = Split(line, ' ');
    std::istringstream iline(split_line[0]);
    mol_number_old = mol_number;
    iline >> mol_number >> mol_name;
    mol_number--;
    atom_names.push_back(split_line[1]);
    if (mol_number_old != mol_number)
      molecule_names.push_back(mol_name);
    index_to_molecule.push_back(mol_number);
    // Find matching molecule
    for (std::vector<Molecule>::const_iterator i_mol = molecules.begin();
         i_mol != molecules.end(); i_mol++) {
      if (mol_name == (*i_mol).name()) {
        mol_match = (*i_mol);
        break;
      }
    }
    // Find matching atom within molecule
    for (std::vector<Atom>::const_iterator i_at = mol_match.begin();
         i_at != mol_match.end(); ++i_at) {
      if (split_line[1] == (*i_at).name()) {
        index_to_mass.push_back((*i_at).mass());
        index_to_sigma.push_back((*i_at).sigma());
        index_to_epsilon.push_back((*i_at).epsilon());
        index_to_charge.push_back((*i_at).charge());
        break;
      }
    }
    for (int i_dim = 0; i_dim<DIMS; i_dim++) {
      positions_(i_atom,i_dim) =
          boost::lexical_cast<double>(split_line[3+i_dim]);
    }
    if (split_line.size() == 9) {
      for (int i_dim = 0; i_dim<DIMS; i_dim++) {
        velocities_(i_atom,i_dim) =
        boost::lexical_cast<double>(split_line[6+i_dim]);
      }
    }
  }
  // Init takes care of initializing the rest of the class matrices
  // and vectors.
  group_size_ = indices_.size();
  Init("all", index_to_mass, index_to_molecule, index_to_sigma,
       index_to_epsilon, index_to_charge, atom_names, molecule_names);
  gro_file.close();
}

void AtomGroup::RemoveAtom(const int index) {
  indices_.erase(indices_.begin() + index);
  atom_names_.erase(atom_names_.begin() + index);
  index_to_mass_.erase(index_to_mass_.begin() + index);
  index_to_molecule_.erase(index_to_molecule_.begin() + index);
  index_to_sigma_.erase(index_to_sigma_.begin() + index);
  index_to_epsilon_.erase(index_to_epsilon_.begin() + index);
  index_to_molecular_index_.erase(
      index_to_molecular_index_.begin() + index);
  positions_.shed_row(index);
  velocities_.shed_row(index);
  group_size_--;
}

void AtomGroup::AddAtom(const Atom atom_to_add, const int molecule_id,
                        const arma::rowvec& position,
                        const arma::rowvec& velocity) {
  atom_names_.push_back(atom_to_add.name());
  index_to_mass_.push_back(atom_to_add.mass());
  index_to_molecule_.push_back(molecule_id);
  index_to_sigma_.push_back(atom_to_add.sigma());
  index_to_epsilon_.push_back(atom_to_add.epsilon());
  index_to_molecular_index_.push_back(num_molecules_);
  // TODO(Zak): Fix to work outside of all_atoms group.
  indices_.push_back(group_size_);
  group_size_++;
  positions_.insert_rows(positions_.n_rows, position);
  velocities_.insert_rows(velocities_.n_rows, velocity);
}

void AtomGroup::FindClusters(const double cutoff_distance_squared,
                             const arma::rowvec& box,
                             Histogram& clusters) const {
  arma::irowvec in_current_cluster;
  arma::irowvec in_any_cluster = arma::zeros<arma::irowvec>(group_size_);
  arma::rowvec dx;
  for (int i_atom = 0; i_atom < group_size_; ++i_atom) {
    if (in_any_cluster(i_atom)) continue;
    in_current_cluster.zeros();
    in_current_cluster(i_atom) = 1;
    in_any_cluster(i_atom) = 1;
    for (int i_other = 0; i_other < group_size_; ++i_other) {
      if (in_any_cluster(i_other)) continue;
      FindDxNoShift(dx, position(i_atom), position(i_other), box);
      double r2 = arma::dot(dx,dx);
      if (r2 > cutoff_distance_squared) continue;
      in_current_cluster(i_other) = 1;
      in_any_cluster(i_other) = 1;
      FindCluster(cutoff_distance_squared, box, dx, in_current_cluster,
                  in_any_cluster, i_other);
    }
    clusters.Add(arma::sum(in_current_cluster));
  }
}

void AtomGroup::FindNearestk(const arma::rowvec& point, const arma::rowvec& box,
                             const int exclude_molecule, const int k,
                             arma::rowvec& dx, arma::rowvec& r2,
                             arma::irowvec& nearest) const {
  for (int i_atom = 0; i_atom < group_size_; ++i_atom) {
    if (index_to_molecule_[i_atom] == exclude_molecule) {
      r2(i_atom) = std::numeric_limits<double>::max();
    } else {
      FindDxNoShift(dx, point, positions_.row(i_atom), box);
      r2(i_atom) = arma::dot(dx,dx);
    }
  }
  arma::uword min_index;
  for (int i = 0; i < k; ++i) {
    r2.min(min_index);
    r2(min_index) = std::numeric_limits<double>::max();
    nearest(i) = min_index;
  }
}

void AtomGroup::FindCluster(const double cutoff_distance_squared,
                             const arma::rowvec& box, arma::rowvec& dx,
                             arma::irowvec& in_current_cluster,
                             arma::irowvec& in_any_cluster,
                             const int start_atom) const {
  for (int i_other = start_atom + 1; i_other < group_size_; ++i_other) {
    if (in_any_cluster(i_other)) continue;
    FindDxNoShift(dx, position(start_atom), position(i_other), box);
    double r2 = arma::dot(dx,dx);
    if (r2 > cutoff_distance_squared) continue;
    in_current_cluster(i_other) = 1;
    in_any_cluster(i_other) = 1;
    FindCluster(cutoff_distance_squared, box, dx, in_current_cluster,
                  in_any_cluster, i_other);
  }
}
