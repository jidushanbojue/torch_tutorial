from Bio.PDB.PDBParser import PDBParser

parser = PDBParser(PERMISSIVE=1)

structure = parser.get_structure('test', '1a1b.pdb')
print('Done')

atoms = structure.get_atoms()
for atom in atoms:
    a = atom
    print(atom)

