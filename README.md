# RCGA for Rastrigin Optimization

Implementacao de um algoritmo genetico codificado em reais (RCGA) para
minimizar a funcao Rastrigin.

## Especificacao atual

Baseado em `references/Exercicio Metaheuristicas MDIOC 2026.pdf`:

- Funcao objetivo: Rastrigin
- Dimensao: 10 variaveis
- Espaco de busca: `[-5, 5]`
- Inicializacao: aleatoria
- Numero de individuos: 30
- Geracoes maximas: `10.000 x dimensao = 100.000`
- Criterio de parada: `f(x) <= 1e-8`

## Parametros bibliograficos

O RCGA usa `eta_c = 1.0` para o cruzamento SBX conforme a referencia
`references/icannga.pdf`, e `eta_m = 20.0` para a mutacao polinomial
conforme `references/978-3-642-35380-2_1.pdf`.

## Como executar uma run

```bash
uv run python src/rcga.py
```

Tambem e possivel controlar os principais parametros:

```bash
uv run python src/rcga.py --seed 42 --max-generations 100000 --zero 1e-8
```

O script imprime a melhor aptidao encontrada, a quantidade de geracoes
executadas, se o criterio de parada foi atingido e o melhor individuo.

## Relatorios

Para executar as 21 runs do exercicio e salvar os resultados dos itens 1 e 2:

```bash
uv run python src/run_experiments.py
```

Os arquivos sao gravados em `reports/`:

- `rcga_convergence.csv`: melhor solucao nas geracoes 1, 50, 100, 200,
  500, 1.000, 5.000, 10.000, 50.000 e 100.000 para cada run.
- `rcga_convergence.svg`: grafico de convergencia com as runs do RCGA.
- `rcga_final_results.csv`: melhor FO final, erro, geracoes executadas e
  melhor individuo de cada run. Este arquivo serve como base para o violin
  plot do erro.
- `rcga_error_violin.svg`: violin plot do erro da funcao objetivo nas runs.
- `rcga_summary.json`: estatisticas `Best`, `Mean`, `Max` e `Std` das
  melhores FOs e dos erros nas 21 runs.
