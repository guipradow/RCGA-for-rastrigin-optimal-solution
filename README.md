# RCGA for Rastrigin Optimization

Implementação de um algoritmo genético de codificação real (RCGA) para
minimizar a função Rastrigin.

## Especificação atual

- Função objetivo: Rastrigin
- Dimensão: 10 variáveis
- Espaço de busca: `[-5, 5]`
- Inicialização: aleatória
- Número de indivíduos: 30
- Número máximo de gerações: `10.000 x dimensão = 100.000`
- Critério de parada: `f(x) <= 1e-8`

## Parâmetros bibliográficos

O RCGA utilizado é baseado no algoritmo apresentado por Prado (2021). O
cruzamento SBX usa `eta_c = 1.0` conforme Deb e Agrawal (1999), e a mutação
polinomial usa `eta_m = 20.0` conforme Deb e Deb (2012).

## Como executar uma run

```bash
uv run python src/rcga.py
```

Também é possível controlar os principais parâmetros:

```bash
uv run python src/rcga.py --seed 42 --max-generations 100000 --zero 1e-8
```

O script imprime a melhor aptidão encontrada, a quantidade de gerações
executadas, se o critério de parada foi atingido e o melhor indivíduo.

## Relatórios

Para executar as 21 runs do exercício e salvar os resultados dos itens 1 e 2:

```bash
uv run python src/run_experiments.py
```

Os arquivos são gravados em `reports/`:

- `rcga_convergence.csv`: melhor solução nas gerações 1, 50, 100, 200,
  500, 1.000, 5.000, 10.000, 50.000 e 100.000 para cada run.
- `rcga_convergence.svg`: gráfico de convergência com as runs do RCGA.
- `rcga_final_results.csv`: melhor FO final, erro, gerações executadas e
  melhor indivíduo de cada run. Este arquivo serve como base para o violin
  plot do erro.
- `rcga_error_violin.svg`: violin plot do erro da função objetivo nas runs.
- `rcga_summary.json`: estatísticas `Best`, `Mean`, `Max` e `Std` das
  melhores FOs e dos erros nas 21 runs.

## Referências

DEB, Kalyanmoy; AGRAWAL, Samir. A niched-penalty approach for constraint
handling in genetic algorithms. Kanpur: Kanpur Genetic Algorithms Laboratory,
Indian Institute of Technology Kanpur, 1999.

DEB, Debayan; DEB, Kalyanmoy. Investigation of mutation schemes in
real-parameter genetic algorithms. In: PANIGRAHI, B. K. et al. (ed.). Swarm,
Evolutionary, and Memetic Computing. Berlin: Springer, 2012. p. 1-8.
(Lecture Notes in Computer Science, v. 7677).

PRADO, Guilherme de Paula. Algoritmo genético de codificação real aplicado à
otimização termoeconômica. 2021. 100 f. Trabalho de Conclusão de Curso
(Graduação) - Universidade Tecnológica Federal do Paraná, Guarapuava, 2021.
