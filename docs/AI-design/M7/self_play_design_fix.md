# 自己対戦学習の設計修正案

## 問題の要約

現在の実装では、自己対戦で2つの独立したネットワークを作成しているが、最終的に保存されるのは主エージェントのネットワークのみ。これにより以下の問題が発生：

1. 対戦相手の学習結果が無駄になる
2. 学習の非対称性が生じる
3. 真の自己対戦学習が実現されない

## 解決策の比較

### 1. 段階的更新アプローチ（推奨）

```python
# 主エージェントのネットワークを基準とし、エピソード毎に対戦相手に同期
def create_self_play_agents(policy_net, value_net, optimizer, algorithm, env):
    # 主エージェント
    main_agent = RLAgent(env, policy_net, value_net, optimizer, algorithm)
    
    # 対戦相手：主エージェントの現在の重みをコピー
    opponent_policy = copy.deepcopy(policy_net)
    opponent_value = copy.deepcopy(value_net)
    
    # 対戦相手は更新しない（frozen）
    for param in opponent_policy.parameters():
        param.requires_grad = False
    for param in opponent_value.parameters():
        param.requires_grad = False
    
    # 対戦相手エージェント（学習なし）
    opponent_agent = RLAgent(env, opponent_policy, opponent_value, None, algorithm)
    
    return main_agent, opponent_agent
```

**メリット**:
- 単一モデルの収束
- 計算効率が良い
- 実装が簡単

**デメリット**:
- 対戦相手が常に少し古いバージョン

### 2. 交互更新アプローチ

```python
# 両エージェントが交互に学習する
def alternating_self_play(policy_net, value_net, optimizer, algorithm, env, episode_num):
    if episode_num % 2 == 0:
        # 偶数エピソード：主エージェントが学習
        learner_agent = RLAgent(env, policy_net, value_net, optimizer, algorithm)
        opponent_agent = RLAgent(env, policy_net, value_net, None, algorithm)  # 学習なし
    else:
        # 奇数エピソード：対戦相手が学習
        learner_agent = RLAgent(env, policy_net, value_net, None, algorithm)  # 学習なし
        opponent_agent = RLAgent(env, policy_net, value_net, optimizer, algorithm)
    
    return learner_agent, opponent_agent
```

### 3. 双方向学習アプローチ

```python
# 両エージェントが同時に学習し、重みを平均化
def bidirectional_self_play(policy_net, value_net, lr, algorithm, env):
    # 独立したオプティマイザーを作成
    main_optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)
    
    # 対戦相手用のネットワーク
    opponent_policy = copy.deepcopy(policy_net)
    opponent_value = copy.deepcopy(value_net)
    opponent_optimizer = optim.Adam(list(opponent_policy.parameters()) + list(opponent_value.parameters()), lr=lr)
    
    main_agent = RLAgent(env, policy_net, value_net, main_optimizer, algorithm)
    opponent_agent = RLAgent(env, opponent_policy, opponent_value, opponent_optimizer, algorithm)
    
    return main_agent, opponent_agent

def merge_networks_after_episode(main_policy, main_value, opp_policy, opp_value):
    # 重みを平均化
    for main_param, opp_param in zip(main_policy.parameters(), opp_policy.parameters()):
        main_param.data = (main_param.data + opp_param.data) / 2
    
    for main_param, opp_param in zip(main_value.parameters(), opp_value.parameters()):
        main_param.data = (main_param.data + opp_param.data) / 2
```

## 推奨実装

**段階的更新アプローチ**を採用することを推奨します：

```python
def create_self_play_opponent(policy_net, value_net, network_config, env):
    """自己対戦用の対戦相手を作成（学習なし）"""
    # 現在の重みをコピー
    opponent_policy = create_policy_network(
        env.observation_space[env.agent_ids[1]],
        env.action_space[env.agent_ids[1]],
        network_config
    )
    opponent_value = create_value_network(
        env.observation_space[env.agent_ids[1]],
        network_config
    )
    
    # 主エージェントの重みをコピー
    opponent_policy.load_state_dict(policy_net.state_dict())
    opponent_value.load_state_dict(value_net.state_dict())
    
    # 勾配計算を無効化
    for param in opponent_policy.parameters():
        param.requires_grad = False
    for param in opponent_value.parameters():
        param.requires_grad = False
    
    return opponent_policy, opponent_value
```

この方法により：
- 主エージェントだけが学習し、その結果が保存される
- 対戦相手は常に主エージェントの現在の状態と対戦
- 計算効率が良く、実装も簡潔
- 単一の最終モデルが得られる