# SIGMODContest2023
## Team Info
| name | email                        | institution |
|------|------------------------------|-------------|
| Jiarui Luo | 11911419@mail.sustech.edu.cn | Southern University of Science and Technology |
## Usage
1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`
5. `./main/test_nndescent data_file output_file K L iter S R gt`

## Notes
1. For competition evaluation, K=100, L=120, iter=14, S=30, R=200. gt is useless for competition evaluation.
2. Some parts of the code has benn hardcoded for the competition, e.g., L2SqrFloatAVX512, IndexGraph::NNDescent
