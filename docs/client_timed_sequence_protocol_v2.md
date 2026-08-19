# GameAgent V2 — execução local de sequências temporizadas

Esta extensão mantém `protocol_version: 2` e preserva o modo `state` para
ações primitivas. Golpes roteirizados — especiais, combos, golpes aéreos,
uppercut, agarrão e rasteira — usam `action_mode: "timed_sequence"`.

O `configure_session_ack` e `/health` anunciam:

```json
{
  "timed_sequences_enabled": true,
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
    "sequence_id": 810,
    "option_name": "Arpão",
    "start_policy": "replace",
    "clock": "monotonic",
    "release_on_complete": true,
    "step_count": 6,
    "total_duration_ms": 405,
    "steps": [
      {"step_index": 1, "commands": [8],    "duration_ms": 75},
      {"step_index": 2, "commands": [],     "duration_ms": 40},
      {"step_index": 3, "commands": [8],    "duration_ms": 75},
      {"step_index": 4, "commands": [8, 2], "duration_ms": 75},
      {"step_index": 5, "commands": [],     "duration_ms": 40},
      {"step_index": 6, "commands": [],  "duration_ms": 100}
    ]
  }
}
```

O exemplo é para o Scorpion olhando para a direita. Quando o servidor detecta
que ele está olhando para a esquerda, envia `9` no lugar de `8`. O cliente
nunca deve espelhar direções.

No passo terminal de um especial, o servidor mantém a última direção junto
do botão (`[8, 2]` no Arpão). Isso reproduz a entrada humana “Trás, Trás e,
sem soltar o segundo Trás, Soco Baixo” e torna o reconhecimento do ZSNES mais
confiável. A mesma regra vale para todos os especiais direcionais.

## Regras obrigatórias do cliente

1. Ao receber `sequence_status: "start"`, cancelar qualquer programa local
   anterior, liberar seus controles e instalar a nova sequência.
2. Executar os passos em uma thread/loop de entrada dedicado, usando relógio
   monotônico. Interface, captura, rede e envio de frames não podem bloquear o
   agendador.
3. Aplicar cada `commands` como o estado completo do controle. Teclas ausentes
   são liberadas e teclas novas são pressionadas atomicamente.
4. Manter o estado por exatamente `duration_ms` e então aplicar o próximo.
   `commands: []` é uma liberação real e nunca pode ser ignorado.
5. Ao terminar, liberar todos os controles porque
   `release_on_complete: true`.
6. `start_policy: "replace"` significa substituir, nunca enfileirar atrás de
   uma sequência antiga.
7. Um `sequence_id` já iniciado é idempotente: mensagem repetida não reinicia
   o golpe.
8. Mensagens posteriores podem trazer o mesmo `sequence_id` com
   `sequence_status: "running"`, sem o objeto `timed_sequence`. Elas são
   telemetria; não reiniciam nem avançam o agendador local.
9. Em modo temporizado, o `commands` do topo tem
   `commands_role: "expected_state_only"`. Ele não controla o emulador; a
   fonte de execução é exclusivamente `timed_sequence.steps`.
10. Se chegar `cancel_pending_sequence: true`, cancelar imediatamente a
    sequência, limpar sua fila, liberar todas as teclas e não executar passos
    restantes.
11. Se chegar uma ação com `action_mode: "state"`, cancelar qualquer sequência
    vencida ou ainda pendente e aplicar normalmente o `commands` do topo.
12. O failsafe continua válido. Ausência de mensagens do servidor além de
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

O cliente deve continuar suportando `action_mode: "state"`. Caso
`timed_sequences_enabled` seja falso ou o servidor não anuncie
`timed_sequence`, deve usar apenas as regras de estado da V2 anterior.
