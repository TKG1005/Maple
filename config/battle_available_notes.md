# battle.available_moves および battle.available_switches のAPI仕様メモ

## available_moves
- 型： List[Move]
- 要素の内容：
    - id (str): 技のID（例: "thunderbolt"）
    - base_power (int): 技の威力
    - accuracy (float): 技の命中率（0〜1の範囲）
    - category (MoveCategory): 技のカテゴリ（例：物理、特殊、変化）
    - type (str): 技のタイプ（例：electric）

- 要素が0になる条件：
    - 全ての技のPPが0になった場合
    - 技が使用禁止の状態（例：こだわりスカーフで別の技を選んだ直後など特殊な状況。ただし、通常は必ず1つ以上の技が使用可能）

## available_switches
- 型： List[Pokemon]
- 要素の内容：
    - species (str): ポケモンの種族名（例: "pikachu"）
    - current_hp_fraction (float): 現在のHPの割合 (0〜1)
    - status (Status): 状態異常（なしの場合はNone）

- 要素が0になる条件：
    - 控えのポケモンが全て瀕死（HPが0）の場合
    - 交代が禁止される技・効果を受けている場合（例:「くろいまなざし」「ありじごく」など）
