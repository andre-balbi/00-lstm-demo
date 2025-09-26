### Configuração Atual de Validação:

  N_SAMPLES = 800
  test_size = 0.2  # 20% para teste final

  ---
  1. Simple Temporal Split

  Passo a Passo:

  - Etapa 1: Separação Inicial (Data Leakage Prevention)
  Dataset (800) → split_for_optimization()
  ├─ Otimização: 640 amostras (80%) ← X_optim, y_optim
  └─ Teste Final: 160 amostras (20%) ← X_test, y_test (ISOLADOS)

  - Etapa 2: Subdivisão para Validação
  Dados de Otimização (640) → temporal_split()
  ├─ Treino: 480 amostras (75% de 640) ← X_train, y_train  
  └─ Validação: 160 amostras (25% de 640) ← X_val, y_val

  - Etapa 3: Otimização (20 trials)
  Para cada trial do Optuna:
  1. Sugere hiperparâmetros (LSTM units, dropout, lr, batch_size)
  2. Cria modelo com esses hiperparâmetros
  3. Treina modelo:
     - Dados: X_train (480), y_train (480)
     - Validação: X_val (160), y_val (160) 
     - Early stopping baseado em val_loss
  4. Calcula RMSE na validação
  5. Retorna RMSE para Optuna otimizar

  - Etapa 4: Modelo Final
  1. Pega melhores hiperparâmetros encontrados
  2. Cria modelo final com esses parâmetros
  3. Treina com TODOS os dados de otimização:
     - X_optim (640), y_optim (640)
     - 100 epochs sem early stopping
  4. Avalia no teste final: X_test (160), y_test (160)

  ---
  2. Walk-Forward Validation

  Passo a Passo:

  - Etapa 1: Separação Inicial (igual)
  Dataset (800) → 640 otimização + 160 teste final

  - Etapa 2: Não há subdivisão prévia
  X_train, X_val = None (não usados)
  Trabalha direto com X_optim (640), y_optim (640)

  - Etapa 3: Otimização (20 trials)
  Para cada trial do Optuna:
  1. Sugere hiperparâmetros
  2. Executa walk_forward_validation():

     Janela móvel de 100 amostras:
     ├─ Ponto 100: treina[0:100] → prediz[100]
     ├─ Ponto 101: treina[1:101] → prediz[101]
     ├─ Ponto 102: treina[2:102] → prediz[102]
     ├─ ...
     └─ Ponto 639: treina[539:639] → prediz[639]

     Total: 540 predições individuais
  3. Calcula RMSE médio das 540 predições
  4. Retorna para Optuna

  - Etapa 4: Modelo Final (igual Simple Temporal)
  1. Treina com todos os dados de otimização (640)
  2. Testa em dados isolados (160)

  ---
  3. Time Series Cross-Validation

  Passo a Passo:

  - Etapa 1: Separação Inicial (igual)
  Dataset (800) → 640 otimização + 160 teste final

  - Etapa 2: Não há subdivisão prévia
  Trabalha direto com X_optim (640), y_optim (640)

  - Etapa 3: Otimização (20 trials)
  Para cada trial do Optuna:
  1. Sugere hiperparâmetros
  2. Executa time_series_cross_validate():

     3 folds crescentes:
     ├─ Fold 1: treina[0:320] → valida[320:427]
     ├─ Fold 2: treina[0:427] → valida[427:533]  
     └─ Fold 3: treina[0:533] → valida[533:640]
     
     Para cada fold:
     - Cria modelo novo
     - Treina com dados do fold
     - Avalia na validação do fold
     - Calcula RMSE individual
     
  3. RMSE final = média dos 3 RMSEs
  4. Retorna para Optuna

  - Etapa 4: Modelo Final (igual)
  1. Treina com todos os dados de otimização (640)
  2. Testa em dados isolados (160)


---


  🔧 PARÂMETROS BÁSICOS DO MODELO

  SEQUENCE_LENGTH = 30

  O que é: Quantos pontos históricos o modelo usa para fazer uma predição
  Exemplo prático:
  - Se você tem dados de vazão diários e SEQUENCE_LENGTH = 30
  - O modelo usa os últimos 30 dias para prever o dia 31
  - Como uma "janela deslizante" de 30 dias

  N_FEATURES = 2

  O que é: Número de variáveis de entrada (características)
  Exemplo prático:
  - Feature 1: Precipitação (mm)
  - Feature 2: Temperatura (°C)
  - O modelo usa essas 2 variáveis para prever a vazão

  ---
  📈 CONFIGURAÇÕES DE DADOS

  N_SAMPLES = 800

  O que é: Quantidade total de amostras sintéticas geradas
  Exemplo prático:
  - 800 dias de dados simulados de vazão
  - Em desenvolvimento: 200 (mais rápido para testar)
  - Em produção: 800 (mais dados = melhor modelo)

  ---
  🎯 OTIMIZAÇÃO OPTUNA

  n_trials = 20

  O que é: Quantas combinações de hiperparâmetros testar
  Exemplo prático:
  Trial 1: 64 neurônios, dropout=0.2, lr=0.005
  Trial 2: 96 neurônios, dropout=0.3, lr=0.001
  ...
  Trial 20: 128 neurônios, dropout=0.1, lr=0.008
  Resultado: Escolhe a melhor combinação automaticamente

  ---
  🏋️ CONFIGURAÇÕES DE TREINAMENTO

  epochs = 50

  O que é: Quantas vezes o modelo vê todos os dados durante otimização
  Exemplo prático:
  - 50 épocas = modelo passa 50 vezes pelos 800 dados
  - Menos épocas = treinamento rápido mas pode não aprender bem
  - Mais épocas = aprende melhor mas demora mais

  final_epochs = 100

  O que é: Épocas para treinar o modelo final (após otimização)
  Por que é maior: Modelo final usa os melhores parâmetros, então pode treinar mais

  ---
  ✅ CONFIGURAÇÕES DE VALIDAÇÃO

  method = 'time_series_cv'

  Opções disponíveis:
  - 'temporal_split': Divide dados em sequência (60% treino, 20% val, 20% teste)
  - 'time_series_cv': Validação cruzada temporal (3 dobras)
  - 'walk_forward': Simula predição em tempo real

  Exemplo time_series_cv:
  Fold 1: [1-200] treino → [201-300] validação
  Fold 2: [1-300] treino → [301-400] validação
  Fold 3: [1-400] treino → [401-500] validação

  test_size = 0.2

  O que é: 20% dos dados reservados para teste final
  Exemplo: De 800 dados, 160 ficam separados e NUNCA são vistos durante treinamento

  cv_splits = 3

  O que é: Número de dobras na validação cruzada
  Mais dobras = validação mais robusta, mas demora mais

  walk_forward_window = 100

  O que é: Tamanho da janela para validação walk-forward
  Exemplo:
  Treina com dados 1-100 → prediz 101
  Treina com dados 2-101 → prediz 102
  Treina com dados 3-102 → prediz 103

  ---
  🧠 LIMITES DOS HIPERPARÂMETROS LSTM

  units_min/max = 32-128

  O que é: Quantidade de neurônios nas camadas LSTM
  Exemplo prático:
  - 32 neurônios: modelo simples, rápido, pode não capturar padrões complexos
  - 128 neurônios: modelo complexo, lento, captura padrões sutis
  - Optuna testa valores entre 32 e 128

  dropout_min/max = 0.1-0.5

  O que é: Prevenção de overfitting
  Exemplo prático:
  - dropout=0.1: "desliga" 10% dos neurônios aleatoriamente
  - dropout=0.5: "desliga" 50% dos neurônios
  - Evita que o modelo "decore" os dados de treino

  lr_min/max = 0.001-0.01

  O que é: Taxa de aprendizado (learning rate)
  Exemplo prático:
  - lr=0.001: aprende devagar mas com mais precisão
  - lr=0.01: aprende rápido mas pode "passar do ponto"
  - Como a velocidade de um carro: muito devagar demora, muito rápido pode bater

  batch_sizes = [16, 32, 64]

  O que é: Quantos exemplos o modelo processa por vez
  Exemplo prático:
  - batch_size=16: processa 16 sequências por vez
  - batch_size=64: processa 64 sequências por vez
  - Maior batch = mais rápido mas usa mais memória

  ---
  ⏹️ CALLBACKS (CONTROLES DE TREINAMENTO)

  early_stopping_patience = 15

  O que é: Para o treinamento se não melhorar por 15 épocas
  Exemplo: Se por 15 épocas seguidas o erro não diminuir, para de treinar

  lr_reduction_patience = 10

  O que é: Reduz learning rate se não melhorar por 10 épocas
  Exemplo: Se estagnar por 10 épocas, reduz lr de 0.01 para 0.005

  lr_reduction_factor = 0.5

  O que é: Fator de redução do learning rate
  Exemplo: Multiplica lr por 0.5 (reduz pela metade)

  ---
  🔬 VALIDAÇÃO ESPECÍFICA

  cv_epochs = 30

  O que é: Épocas usadas durante validação cruzada (menor que o treino final)
  Por que menos: Validação cruzada testa muitas combinações, então usa menos épocas para ser mais rápido

  walk_forward_epochs = 10

  O que é: Épocas para cada janela do walk-forward
  Por que tão pouco: Walk-forward treina muitos mini-modelos, então cada um treina rapidamente

  walk_forward_batch_size = 16

  O que é: Batch size específico para walk-forward
  Menor porque: Janelas menores de dados precisam de batch menor