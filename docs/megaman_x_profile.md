# Perfil Mega Man X

`megaman_x` representa a família Mega Man X. Os aliases `megaman_x1`,
`megaman_x2` e `megaman_x3` selecionam o mesmo perfil inicial. A infraestrutura
compartilhada de GameAgent continua idêntica; ações, estado, recompensa,
checkpoints e replay serão próprios deste perfil.

## Coleta visual inicial

Antes de existir treino, execute `bash scripts/run_megaman_x_capture.sh`. O
servidor aceita JPEGs, valida/decodifica cada imagem e responde com
`action_name: "capture_only"` e `commands: []`; portanto, ele não interfere
nos controles. Com `GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS=3000`, salva uma
imagem a cada três segundos em `/workspace/gameagent/megaman_x_captures`.
Essas capturas servem para localizar vida, barra de chefe, inimigos, morte e
progresso de fase antes de implementar o adaptador de estado.

## Layout temporário de controles

O cliente Mega Man deve usar os códigos já definidos pelo protocolo V2:

| Código | Controle Mega Man |
| --- | --- |
| `1` | X / dash |
| `2` | B / tiro e carga |
| `3` | A / pulo |
| `6` | cima |
| `7` | baixo |
| `8` | esquerda |
| `9` | direita |

O cliente deve confirmar no `action_ack` os comandos que realmente aplicou.
Uma sequência é cancelável por uma ação com `cancel_pending_sequence: true` ou
por `release_controls: true`, exatamente como no perfil de Mortal Kombat.

## Catálogo inicial

| ID | Estados (códigos simultâneos) | Tempo |
| --- | --- | --- |
| `shoot` | `[2]` | 75 ms |
| `charge_shot` | `[2]` | 600 ms |
| `super_charge_shot` | `[2]` | 1200 ms |
| `double_charge_shot` | `[2]` | 1800 ms |
| `jump` | `[3]` | 100 ms |
| `air_dash` | `[3]`, `[1]` | 90 ms por estado |
| `long_jump` | `[2,3]` | 350 ms |
| `wall_climb_right` | `[9,2]`, `[]`, `[9,2]`, `[]`, `[9,2]` | 90 ms por estado e 35 ms neutro |
| `wall_climb_left` | `[8,2]`, `[]`, `[8,2]`, `[]`, `[8,2]` | 90 ms por estado e 35 ms neutro |

Todo programa encerra com liberação de 100 ms. Os tempos são iniciais e devem
ser refinados por captura/validação no emulador alvo.

## Próxima integração

O perfil ainda não executa treinamento. Antes disso o cliente deverá transmitir
um estado semântico de plataforma: posição horizontal/vertical do jogador,
velocidade ou direção, chão/parede/ar, vida, progresso de fase, inimigos,
dano, morte e conclusão de fase. Esse adaptador substituirá integralmente o
detector de combate de Mortal Kombat.
