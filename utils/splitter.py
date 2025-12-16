import pandas as pd
import os

# Ler o arquivo CSV
df = pd.read_csv('results/results.csv')

# Calcular o tamanho de cada parte
total_rows = len(df)
chunk_size = total_rows // 3

# Criar o diretório de saída se não existir
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Dividir e salvar em 3 arquivos
df.iloc[:chunk_size].to_csv(f'{output_dir}/results_part1.csv', index=False)
df.iloc[chunk_size:2*chunk_size].to_csv(f'{output_dir}/results_part2.csv', index=False)
df.iloc[2*chunk_size:].to_csv(f'{output_dir}/results_part3.csv', index=False)

print(f"Arquivo dividido em 3 partes:")
print(f"- {output_dir}/results_part1.csv: {chunk_size} linhas")
print(f"- {output_dir}/results_part2.csv: {chunk_size} linhas")
print(f"- {output_dir}/results_part3.csv: {total_rows - 2*chunk_size} linhas")