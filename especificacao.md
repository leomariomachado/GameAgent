# Alteração do cliente para o protocolo GameAgent WebSocket v2

## Objetivo

Substituir o fluxo HTTP atual por uma conexão WebSocket persistente capaz de:

- capturar frames reais do jogo;
- enviar JPEGs continuamente ao RunPod;
- receber ações aproximadamente 20 vezes por segundo;
- interpretar cada ação como o estado completo dos controles;
- manter e liberar corretamente os botões;
- executar botões simultâneos, diagonais e sequências;
- liberar todos os controles em caso de timeout ou desconexão;
- reconectar automaticamente sem travar a interface.

---

## 1. Acessos

### WebSocket

```text
wss://54x9n8p474bqla-8080.proxy.runpod.net/ws
```

### Health check

```text
https://54x9n8p474bqla-8080.proxy.runpod.net/health
```

### Autenticação

Usar o mesmo token configurado no RunPod como:

```text
GAMEAGENT_API_TOKEN
```

Enviar no upgrade WebSocket:

```http
Authorization: Bearer SEU_TOKEN
```

Exemplo com Qt:

```cpp
QNetworkRequest request(
    QUrl("wss://54x9n8p474bqla-8080.proxy.runpod.net/ws")
);

request.setRawHeader(
    "Authorization",
    QByteArray("Bearer ") + apiToken
);

webSocket.open(request);
```

Não colocar o token na URL nem nos logs.

---

## 2. Remover o fluxo HTTP em lote

O cliente não deve mais:

- acumular 30 frames;
- converter todos para Base64;
- criar um JSON grande;
- fazer um POST HTTP;
- esperar uma resposta para então reiniciar a captura.

O novo fluxo deve ser contínuo:

```text
capturar frame
→ codificar como JPEG
→ enviar mensagem WebSocket binária
→ continuar capturando
```

O recebimento das ações deve acontecer em paralelo.

O fluxo HTTP atual em `src/MainWindow.cpp` deve ser substituído pelo WebSocket persistente.

---

## 3. Taxas

### Captura e envio

```text
15 frames por segundo
```

Intervalo aproximado:

```text
67 ms
```

### Ações recebidas

```text
20 decisões por segundo
```

Intervalo aproximado:

```text
50 ms
```

As taxas são independentes. É normal que duas ações consecutivas utilizem o mesmo frame mais recente.

---

## 4. Captura

Configuração inicial:

```text
Resolução aproximada: 512×448
Formato: JPEG
Qualidade: 80
Taxa: 15 fps
```

Capturar apenas a área útil do jogo.

Não incluir:

- bordas da janela;
- barra de título;
- cursor;
- desktop;
- overlays desnecessários.

Cada captura deve gerar um JPEG novo. Não reutilizar uma imagem fixa.

---

## 5. Fila de frames

O cliente não deve acumular frames antigos.

Deve existir no máximo:

```text
1 frame aguardando transmissão
```

Se um frame novo ficar pronto enquanto outro ainda aguarda envio:

```text
descartar o frame pendente antigo
e manter somente o mais recente
```

A prioridade é baixa latência, não garantir a entrega de todos os frames.

Nunca criar uma fila crescente de imagens.

---

## 6. Identificação

Cada conexão deve possuir um contador:

```cpp
quint64 nextFrameId = 1;
```

Para cada frame:

```cpp
const quint64 frameId = nextFrameId++;
```

O contador deve:

- aumentar monotonicamente;
- não repetir dentro da conexão;
- poder voltar para 1 depois de reconectar.

---

## 7. Timestamp

Cada frame deve conter o horário da captura em Unix epoch milliseconds:

```cpp
const qint64 timestampMs =
    QDateTime::currentDateTimeUtc().toMSecsSinceEpoch();
```

O timestamp deve ser obtido no momento da captura, antes da codificação JPEG.

---

## 8. Formato binário enviado

Cada JPEG deve ser enviado em uma mensagem WebSocket binária separada.

Formato:

| Offset | Tamanho | Tipo | Campo |
|---:|---:|---|---|
| 0 | 4 bytes | ASCII | `GAF1` |
| 4 | 8 bytes | uint64 big-endian | `frame_id` |
| 12 | 8 bytes | int64 big-endian | `timestamp_ms` |
| 20 | 4 bytes | uint32 big-endian | tamanho do JPEG |
| 24 | N bytes | binário | JPEG |

Representação:

```text
"GAF1"
+ frame_id
+ timestamp_ms
+ jpeg_size
+ jpeg
```

Todos os números devem ser big-endian.

Exemplo com Qt:

```cpp
QByteArray makeFrameMessage(
    quint64 frameId,
    qint64 timestampMs,
    const QByteArray& jpeg
) {
    QByteArray payload;

    {
        QDataStream stream(&payload, QIODevice::WriteOnly);
        stream.setByteOrder(QDataStream::BigEndian);

        stream.writeRawData("GAF1", 4);
        stream << frameId;
        stream << timestampMs;
        stream << static_cast<quint32>(jpeg.size());
    }

    payload.append(jpeg);
    return payload;
}
```

Envio:

```cpp
const QByteArray message = makeFrameMessage(
    frameId,
    timestampMs,
    jpeg
);

webSocket.sendBinaryMessage(message);
```

Não usar Base64 no fluxo WebSocket.

Não colocar mais de um JPEG na mesma mensagem.

---

## 9. Formato recebido

O servidor envia mensagens WebSocket de texto contendo JSON:

```json
{
  "protocol_version": 2,
  "type": "action",
  "action_mode": "state",
  "commands": [9, 3],
  "decision_id": 123,
  "frame_id": 456,
  "frame_ids": [449, 450, 451, 452, 453, 454, 455, 456],
  "timestamp_ms": 1784740000170,
  "control_interval_ms": 50,
  "failsafe_timeout_ms": 200,
  "metrics": {
    "frame_latency_ms": 47,
    "server_frame_age_ms": 18,
    "inference_ms": 0.018,
    "buffer_size": 32,
    "frames_received": 456,
    "frames_dropped": 424,
    "decode_errors": 0
  }
}
```

O cliente deve processar somente mensagens com:

```json
{
  "protocol_version": 2,
  "type": "action",
  "action_mode": "state"
}
```

Mensagens inválidas devem ser registradas e ignoradas.

Uma mensagem inválida não deve liberar imediatamente os controles. O failsafe cuidará disso se nenhuma ação válida chegar.

---

## 10. Mapeamento

| Valor | Controle |
|---:|---|
| 0 | Y |
| 1 | X |
| 2 | B |
| 3 | A |
| 4 | L |
| 5 | R |
| 6 | Cima |
| 7 | Baixo |
| 8 | Esquerda |
| 9 | Direita |

Os valores devem ser tratados como um conjunto sem ordem.

---

## 11. Regra mais importante: `commands` é estado

O vetor não representa cliques.

Ele representa todos os controles que devem estar pressionados naquele momento.

Exemplo:

```json
{"commands":[9,3]}
```

Significa:

```text
manter Direita e A pressionados
```

A próxima ação substitui completamente a anterior.

Exemplo:

```text
Ação 1: [9,3]
Ação 2: [9]
Ação 3: []
```

Interpretação:

```text
Ação 1: pressionar Direita e A
Ação 2: manter Direita e soltar A
Ação 3: soltar tudo
```

---

## 12. Estado interno

Manter no cliente:

```cpp
QSet<int> currentCommands;
quint64 lastDecisionId = 0;
```

Ao receber uma ação válida:

```cpp
void applyAction(
    quint64 decisionId,
    const QSet<int>& newCommands
) {
    if (decisionId <= lastDecisionId) {
        return;
    }

    const QSet<int> commandsToRelease =
        currentCommands - newCommands;

    const QSet<int> commandsToPress =
        newCommands - currentCommands;

    releaseCommands(commandsToRelease);
    pressCommands(commandsToPress);

    currentCommands = newCommands;
    lastDecisionId = decisionId;

    restartFailsafeTimer();
}
```

Somente comandos novos devem gerar `keyDown`.

Somente comandos removidos devem gerar `keyUp`.

Comandos que permanecem no vetor não devem gerar eventos repetidos.

---

## 13. Vetor vazio

Isto é uma ação válida:

```json
{
  "commands": []
}
```

Significa:

```text
soltar todos os controles
```

Não interpretar como erro, resposta ausente ou “manter ação anterior”.

---

## 14. Botões simultâneos

Exemplo:

```json
{
  "commands": [9, 3, 5]
}
```

Significa:

```text
Direita + A + R pressionados simultaneamente
```

Aplicar todas as mudanças no mesmo callback de ação, antes de retornar ao event loop.

A diferença de microssegundos entre eventos de teclado será percebida pelo emulador como simultânea no próximo frame.

---

## 15. Diagonais

As diagonais são representadas por duas direções:

```text
[6,9] = Cima + Direita
[6,8] = Cima + Esquerda
[7,9] = Baixo + Direita
[7,8] = Baixo + Esquerda
```

O cliente não deve rejeitar duas direções perpendiculares.

Combinações opostas, como Cima+Baixo, podem aparecer no mock. Devem ser toleradas durante o teste.

---

## 16. Sequências

Sequências são formadas por vários estados consecutivos.

Exemplo de quarto de lua para a direita seguido de A:

```text
t=0 ms:   [7]
t=50 ms:  [7,9]
t=100 ms: [9]
t=150 ms: [9,3]
t=200 ms: []
```

O cliente não precisa reconhecer “meia-lua” ou “quarto de lua”.

Ele apenas aplica corretamente cada estado recebido.

Não criar macros nem filas locais de sequências nesta versão.

---

## 17. Toques repetidos

Para apertar A duas vezes, o servidor deve produzir:

```text
[3] → [] → [3] → []
```

O cliente interpreta:

```text
pressionar A
soltar A
pressionar A
soltar A
```

Isto:

```text
[3] → [3]
```

significa manter A pressionado, não apertar duas vezes.

---

## 18. Failsafe

Cada ação informa:

```json
{
  "failsafe_timeout_ms": 200
}
```

O cliente deve reiniciar um timer somente depois de receber e aplicar uma ação válida.

Se nenhuma nova ação válida chegar em 200 ms:

```text
soltar todos os controles
limpar currentCommands
```

Não desconectar necessariamente o WebSocket; apenas entrar em estado neutro.

O timeout não deve ser 50 ms, porque pequenas variações de rede poderiam causar liberações indevidas.

---

## 19. Desconexão

Ao detectar desconexão:

1. Soltar todos os controles imediatamente.
2. Limpar `currentCommands`.
3. Limpar o frame pendente.
4. Invalidar callbacks da conexão antiga.
5. Reiniciar `lastDecisionId`.
6. Iniciar reconexão automática.

Função obrigatória:

```cpp
void releaseAllControls() {
    releaseCommands(currentCommands);
    currentCommands.clear();
}
```

Ela também deve ser chamada:

- ao fechar o programa;
- ao parar a captura;
- ao trocar de emulador;
- em erro fatal;
- antes de destruir a janela principal.

---

## 20. Reconexão

Usar backoff:

```text
1 s → 2 s → 4 s → 8 s → máximo de 10 s
```

Depois de conectar com sucesso:

```text
backoff volta para 1 s
frame_id volta para 1
lastDecisionId volta para 0
```

Manter apenas uma conexão WebSocket ativa.

---

## 21. Ping e conexão ociosa

Enviar ping WebSocket a cada 20–30 segundos.

A biblioteca deve responder automaticamente a pings do servidor.

A ausência de pong ou o fechamento da conexão deve iniciar o fluxo de reconexão.

---

## 22. Organização recomendada

Separar em componentes:

### `FrameCapture`

- captura a janela;
- recorta a área do jogo;
- gera JPEG;
- usa timer de 67 ms.

### `GameAgentWebSocketClient`

- abre o WebSocket;
- adiciona autenticação;
- cria o envelope binário;
- envia frames;
- recebe JSON;
- trata ping, desconexão e reconexão.

### `ControllerState`

- mantém `currentCommands`;
- calcula pressões e liberações;
- aplica `keyDown` e `keyUp`;
- executa o failsafe;
- libera tudo na desconexão.

A captura não deve depender da chegada de uma resposta.

A aplicação da ação não deve esperar a próxima captura.

---

## 23. Métricas locais recomendadas

Registrar:

```text
frames_captured
frames_sent
frames_replaced_before_send
actions_received
actions_ignored
last_decision_id
websocket_reconnects
failsafe_releases
```

Também registrar do servidor:

```text
frame_id
server_frame_age_ms
inference_ms
decode_errors
frames_dropped
```

Não registrar o token nem o conteúdo completo dos JPEGs.

---

## 24. Critérios de aceite

A implementação estará correta quando:

1. Conectar em `wss://54x9n8p474bqla-8080.proxy.runpod.net/ws`.
2. Enviar autenticação durante o upgrade.
3. Capturar aproximadamente 15 fps.
4. Enviar cada JPEG como mensagem binária `GAF1`.
5. Não usar Base64.
6. Manter no máximo um frame pendente.
7. Receber ações de protocolo versão 2.
8. Rejeitar `decision_id` antigo ou repetido.
9. Interpretar `commands` como estado completo.
10. Não repetir `keyDown` para uma tecla já pressionada.
11. Soltar apenas teclas ausentes na nova ação.
12. Interpretar `[]` como soltar tudo.
13. Executar corretamente botões simultâneos e diagonais.
14. Executar sequências por estados sucessivos.
15. Liberar tudo após 200 ms sem ação válida.
16. Liberar tudo imediatamente na desconexão.
17. Reconectar automaticamente.
18. Manter a interface responsiva.

---

## 25. Testes obrigatórios

### Manutenção

Entrada:

```text
[9,3] → [9,3] → [9]
```

Esperado:

```text
pressionar Direita+A
nenhum novo evento
soltar A e manter Direita
```

### Vetor vazio

Entrada:

```text
[9] → []
```

Esperado:

```text
pressionar Direita
soltar Direita
```

### Toque repetido

Entrada:

```text
[3] → [] → [3] → []
```

Esperado:

```text
dois apertos distintos de A
```

### Botões simultâneos

Entrada:

```text
[9,3,5]
```

Esperado:

```text
Direita+A+R mantidos simultaneamente
```

### Diagonal

Entrada:

```text
[7,9]
```

Esperado:

```text
Baixo+Direita
```

### Ação antiga

Entrada:

```text
decision_id=10
decision_id=9
```

Esperado:

```text
aplicar 10
ignorar 9
```

### Failsafe

Entrada:

```text
[9,3]
nenhuma resposta por mais de 200 ms
```

Esperado:

```text
soltar Direita e A
```

### Desconexão

Entrada:

```text
[9,3]
desconectar WebSocket
```

Esperado:

```text
soltar tudo imediatamente
iniciar reconexão
```