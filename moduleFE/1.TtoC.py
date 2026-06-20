import pandas as pd

def txt_to_csv(txt_file_path, csv_file_path):

    data = pd.read_csv(txt_file_path, sep='\s+')

    data.to_csv(csv_file_path, index=False)

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

txt_to_csv(input_ilm_indel_train_file_txt, output_ilm_indel_train_file_csv)
txt_to_csv(input_ilm_indel_valid_file_txt, output_ilm_indel_valid_file_csv)
txt_to_csv(input_ilm_indel_test_file_txt, output_ilm_indel_test_file_csv)
txt_to_csv(input_ion_indel_train_file_txt, output_ion_indel_train_file_csv)
txt_to_csv(input_ion_indel_valid_file_txt, output_ion_indel_valid_file_csv)
txt_to_csv(input_ion_indel_test_file_txt, output_ion_indel_test_file_csv)
txt_to_csv(input_ilm_snp_train_file_txt, output_ilm_snp_train_file_csv)
txt_to_csv(input_ilm_snp_valid_file_txt, output_ilm_snp_valid_file_csv)
txt_to_csv(input_ilm_snp_test_file_txt, output_ilm_snp_test_file_csv)
txt_to_csv(input_ion_snp_train_file_txt, output_ion_snp_train_file_csv)
txt_to_csv(input_ion_snp_valid_file_txt, output_ion_snp_valid_file_csv)
txt_to_csv(input_ion_snp_test_file_txt, output_ion_snp_test_file_csv)

print("文件转换成功并保存")