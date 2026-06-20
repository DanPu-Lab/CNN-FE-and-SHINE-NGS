import pandas as pd

def check(txt_file_path, csv_file_path):
    data1 = pd.read_csv(txt_file_path, sep='\s+')
    data2 = pd.read_csv(csv_file_path)

    if data1.equals(data2):
        return 1
    else:
        return 0

input_ilm_indel_train_file_txt = '../datasets/ILM_INDEL_Training.txt'
output_ilm_indel_train_file_csv = 'ILM_INDEL_Train.csv'

input_ilm_indel_valid_file_txt = '../datasets/ILM_INDEL_Validation1.txt'
output_ilm_indel_valid_file_csv = 'ILM_INDEL_Valid.csv'

input_ilm_indel_test_file_txt = '../datasets/ILM_INDEL_Test.txt'
output_ilm_indel_test_file_csv = 'ILM_INDEL_Test.csv'

input_ilm_snp_train_file_txt = '../datasets/ILM_SNP_Training.balance.txt'
output_ilm_snp_train_file_csv = 'ILM_SNP_Train.csv'

input_ilm_snp_valid_file_txt = '../datasets/ILM_SNP_Validation1.txt'
output_ilm_snp_valid_file_csv = 'ILM_SNP_Valid.csv'

input_ilm_snp_test_file_txt = '../datasets/ILM_SNP_Test.txt'
output_ilm_snp_test_file_csv = 'ILM_SNP_Test.csv'

input_ion_indel_train_file_txt = '../datasets/ION_INDEL_Training.txt'
output_ion_indel_train_file_csv = 'ION_INDEL_Train.csv'

input_ion_indel_valid_file_txt = '../datasets/ION_INDEL_Validation1.txt'
output_ion_indel_valid_file_csv = 'ION_INDEL_Valid.csv'

input_ion_indel_test_file_txt = '../datasets/ION_INDEL_Test.txt'
output_ion_indel_test_file_csv = 'ION_INDEL_Test.csv'

input_ion_snp_train_file_txt = '../datasets/ION_SNP_Training.balance.txt'
output_ion_snp_train_file_csv = 'ION_SNP_Train.csv'

input_ion_snp_valid_file_txt = '../datasets/ION_SNP_Validation1.txt'
output_ion_snp_valid_file_csv = 'ION_SNP_Valid.csv'

input_ion_snp_test_file_txt = '../datasets/ION_SNP_Test.txt'
output_ion_snp_test_file_csv = 'ION_SNP_Test.csv'

a = check(input_ilm_indel_train_file_txt, output_ilm_indel_train_file_csv)
b = check(input_ilm_indel_valid_file_txt, output_ilm_indel_valid_file_csv)
c = check(input_ilm_indel_test_file_txt, output_ilm_indel_test_file_csv)
d = check(input_ion_indel_train_file_txt, output_ion_indel_train_file_csv)
e = check(input_ion_indel_valid_file_txt, output_ion_indel_valid_file_csv)
f = check(input_ion_indel_test_file_txt, output_ion_indel_test_file_csv)
g = check(input_ilm_snp_train_file_txt, output_ilm_snp_train_file_csv)
h = check(input_ilm_snp_valid_file_txt, output_ilm_snp_valid_file_csv)
i = check(input_ilm_snp_test_file_txt, output_ilm_snp_test_file_csv)
j = check(input_ion_snp_train_file_txt, output_ion_snp_train_file_csv)
k = check(input_ion_snp_valid_file_txt, output_ion_snp_valid_file_csv)
l = check(input_ion_snp_test_file_txt, output_ion_snp_test_file_csv)

if(a + b + c + d + e + f + g + h + i + j + k + l== 12):
    print("数据一致")