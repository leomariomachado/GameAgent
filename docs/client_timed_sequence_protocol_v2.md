# GameAgent V2 — sequências temporizadas com estado exato (revisão 2)

Esta extensão mantém `protocol_version: 2` e preserva o modo `state` para
ações primitivas. Golpes roteirizados — especiais, combos, golpes aéreos,
uppercut, agarrão e rasteira — usam `action_mode: "timed_sequence"`.

A revisão 2 remove qualquer herança implícita entre passos. Cada `commands`
é o estado completo do controle naquele intervalo. Isso permite representar
sem ambiguidade tanto um especial cuja direção continua pressionada junto ao
botão quanto um salto em que a direção deve ser liberada antes do chute.

O `configure_session_ack` e `/health` anunciam:

```json
{
  "timed_sequences_enabled": true,
  "timed_sequence_version": 2,
  "timed_sequence_step_semantics": "exact_controller_state",
  "supported_action_modes": ["state", "timed_sequence"],
  "sequence_input_hold_ms": 75,
  "sequence_neutral_hold_ms": 40,
  "sequence_final_release_ms": 100,
  "sequence_actionable_ticks": 2
}
```

## Início de uma sequência

```json
{
  "protocol_version": 2,
  "type": "action",
  "action_mode": "timed_sequence",
  "commands_role": "expected_state_only",
  "decision_id": 810,
  "sequence_id": 810,
  "sequence_status": "start",
  "cancel_pending_sequence": false,
  "rl_option": "Arpão",
  "timed_sequence": {
    "version": 2,
    "sequence_id": 810,
    "option_name": "Arpão",
    "start_policy": "replace",
    "step_semantics": "exact_controller_state",
    "transition_policy": "replace_state",
    "clock": "monotonic",
    "release_on_complete": true,
    "step_count": 6,
    "total_duration_ms": 405,
    "steps": [
      {"step_index": 1, "commands": [8],    "state_kind": "direction", "apply": "replace_state", "duration_ms": 75},
      {"step_index": 2, "commands": [],     "state_kind": "neutral", "apply": "replace_state", "duration_ms": 40},
      {"step_index": 3, "commands": [8],    "state_kind": "direction", "apply": "replace_state", "duration_ms": 75},
      {"step_index": 4, "commands": [8, 2], "state_kind": "direction_button_chord", "apply": "replace_state", "duration_ms": 75},
      {"step_index": 5, "commands": [],     "state_kind": "neutral", "apply": "replace_state", "duration_ms": 40},
      {"step_index": 6, "commands": [],     "state_kind": "neutral", "apply": "replace_state", "duration_ms": 100}
    ]
  }
}
```

O exemplo é para o Scorpion olhando para a direita. Quando o servidor detecta
que ele está olhando para a esquerda, envia `9` no lugar de `8`. O cliente
nunca deve espelhar direções.

Quando o servidor aplicar a versão espelhada, o nome de exibição terá o
sufixo ` (E)`, por exemplo `Arpão (E)`. O campo `rl_option_base` preserva o
nome estável `Arpão` e `rl_commands_mirrored: true` continua sendo a fonte
estruturada dessa informação. O sufixo é apenas visual e não muda a identidade
da ação para treinamento ou telemetria.

No passo terminal de um especial, o próprio programa declara a última direção
junto do botão (`[8, 2]` no Arpão). Não existe uma regra global no transporte
que acrescente essa direção. Por exemplo, o salto chutando pode conter
`[9, 1]` seguido por `[1]`: no segundo passo o cliente deve soltar `9` e manter
somente `1`. Rasteira e agarrão já declaram seus acordes completos em cada
passo. Assim cada família conserva sua natureza sem heurística compartilhada.

## Regras obrigatórias do cliente

1. Ao receber `sequence_status: "start"`, cancelar qualquer programa local
   anterior, liberar seus controles e instalar a nova sequência.
2. Executar os passos em uma thread/loop de entrada dedicado, usando relógio
   monotônico. Interface, captura, rede e envio de frames não podem bloquear o
   agendador.
3. Exigir `timed_sequence.version == 2`,
   `step_semantics == "exact_controller_state"` e
   `transition_policy == "replace_state"`.
4. Aplicar cada `commands` como o estado completo do controle. Antes de cada
   passo, substituir o estado anterior: teclas ausentes são liberadas e teclas
   novas são pressionadas atomicamente. Nunca herdar ou acrescentar direção.
5. Manter o estado por exatamente `duration_ms` e então aplicar o próximo.
   `commands: []` é uma liberação real e nunca pode ser ignorado.
6. Ao terminar, liberar todos os controles porque
   `release_on_complete: true`.
7. `start_policy: "replace"` significa substituir, nunca enfileirar atrás de
   uma sequência antiga.
8. Um `sequence_id` já iniciado é idempotente: mensagem repetida não reinicia
   o golpe.
9. Mensagens posteriores podem trazer o mesmo `sequence_id` com
   `sequence_status: "running"`, sem o objeto `timed_sequence`. Elas são
   telemetria; não reiniciam nem avançam o agendador local.
10. Em modo temporizado, o `commands` do topo tem
   `commands_role: "expected_state_only"`. Ele não controla o emulador; a
   fonte de execução é exclusivamente `timed_sequence.steps`.
11. Se chegar `cancel_pending_sequence: true`, cancelar imediatamente a
    sequência, limpar sua fila, liberar todas as teclas e não executar passos
    restantes.
12. Se chegar uma ação com `action_mode: "state"`, cancelar qualquer sequência
    vencida ou ainda pendente e aplicar normalmente o `commands` do topo.
13. O failsafe continua válido. Ausência de mensagens do servidor além de
    `failsafe_timeout_ms` cancela a sequência e libera os controles.

## ACK

O cliente continua enviando um `action_ack` para cada mensagem `action`.
`applied_commands` deve representar as teclas realmente aplicadas pelo
agendador no instante do ACK, não copiar cegamente o `commands` recebido:

```json
{
  "protocol_version": 2,
  "type": "action_ack",
  "decision_id": 811,
  "applied_commands": [8],
  "applied_timestamp_ms": 1787100000123
}
```

Durante uma sequência local, o estado real pode diferir do estado de
diagnóstico do servidor. O servidor marca essas comparações como
`commands_advisory: true` e não as trata como erro.

## Precisão e diagnóstico

- Aplicar o primeiro passo imediatamente após aceitar a sequência.
- Buscar erro de agendamento menor ou igual a um frame do emulador
  (aproximadamente 16,7 ms a 60 Hz).
- Registrar `sequence_id`, `step_index`, tempo planejado, tempo aplicado,
  atraso e conjunto efetivamente injetado.
- Não usar temporizadores da interface gráfica nem eventos de teclado do SO se
  existir acesso direto ao estado do controle do emulador.
- Na interface, mostrar `rl_option_step/rl_option_steps` e distinguir
  `start`, `running`, `release` e cooldown. O nome do golpe sozinho não indica
  que uma tecla está sendo pressionada naquele instante.

## Compatibilidade

O cliente deve continuar suportando `action_mode: "state"`. A revisão 2 não
altera o `protocol_version` externo. Clientes que ainda só conhecem a primeira
revisão temporizada devem recusar `timed_sequence.version: 2` e liberar o
controle, pois executar a sequência com herança de direção produziria golpes
incorretos. Caso `timed_sequences_enabled` seja falso, deve usar apenas as
regras de estado da V2 anterior.
