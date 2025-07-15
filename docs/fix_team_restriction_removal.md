# Team Restriction System Removal - Fix Documentation

## 🚨 Problem Summary

The team restriction system was causing critical action mask failures during Pokemon battles, leading to training crashes when all actions became disabled.

## 🔍 Root Cause Analysis

### Problem Flow
1. **Initial Setup**: `_selected_species` tracks Pokemon selected during team preview
2. **Form Changes**: Pokemon like Mimikyu change forms during battle
   - Initial: `'mimikyu'` (registered in `_selected_species`)
   - After damage: `'mimikyubusted'` (not recognized as same Pokemon)
3. **Team Restriction Logic**: System incorrectly treats form-changed Pokemon as "unselected"
4. **Action Mask Failure**: Valid switch actions get disabled, leading to all-zero masks

### Specific Example from Logs
```
DEBUG: player_0 selected_species = {'irontreads', 'weezinggalar', 'mimikyu'}
DEBUG: player_0 switch idx=8, pkmn.species=mimikyubusted, in_selected=False
DEBUG: player_0 disabling switch idx=8 (species mimikyubusted not in selected)
```

**Result**: Action 8 (switch to Mimikyu-Busted) disabled despite being valid battle action.

## 🛠️ Solution: Complete System Removal

### Why Complete Removal?
1. **Form Change Complexity**: Pokemon have numerous forms (Mimikyu-Busted, Darmanitan-Zen, etc.)
2. **Maintenance Burden**: Mapping all form changes is error-prone
3. **Battle Engine Reliability**: `poke-env` already handles valid switches correctly
4. **Training Priority**: Stability over strict team preview enforcement

### Changes Made

#### 1. Data Structure Cleanup
```python
# REMOVED from __init__():
self._selected_species: dict[str, set[str]] = {
    agent_id: set() for agent_id in self.agent_ids
}
```

#### 2. Action Mask Generation Fix
```python
# REMOVED from _compute_all_masks():
selected = self._selected_species.get(pid)
if selected:
    for idx, (atype, sub_idx, disabled) in mapping.items():
        if atype == "switch" and not disabled:
            # Complex form checking logic
            if pkmn.species not in selected:
                mask[idx] = 0  # ← This was causing the bug
```

#### 3. Team Preview Processing
```python
# REMOVED from step():
indices = [int(x) - 1 for x in re.findall(r"\d", act)]
roster = self._team_rosters.get(agent_id, [])
self._selected_species[agent_id] = {
    roster[i] for i in indices if 0 <= i < len(roster)
}
```

#### 4. Deprecated Method Cleanup
```python
# REMOVED from get_action_mask():
selected = self._selected_species.get(player_id)
if selected:
    for idx, detail in mapping.items():
        if (detail.get("type") == "switch" and 
            detail.get("id") not in selected):
            mask[idx] = 0
```

## 📊 Impact Analysis

### Benefits
- ✅ **Eliminates Action Mask Failures**: No more all-zero action masks
- ✅ **Handles All Pokemon Forms**: Works with any form changes
- ✅ **Simplifies Codebase**: Removes complex restriction logic
- ✅ **Improves Training Stability**: Prevents common crash scenarios
- ✅ **Maintains Battle Validity**: poke-env still enforces valid switches

### Tradeoffs
- ❌ **Team Preview Enforcement**: No longer restricts to selected Pokemon
- ❌ **Competitive Accuracy**: Less faithful to official Pokemon rules

### Technical Impact
- **Lines of Code**: -32 lines removed
- **Complexity**: Reduced cognitive load in action mask generation
- **Performance**: Slightly faster mask computation
- **Maintenance**: Eliminates form change mapping requirements

## 🧪 Testing Recommendations

### Test Cases to Verify
1. **Form Change Pokemon**: Mimikyu, Darmanitan-Zen, etc.
2. **Multi-Battle Training**: Ensure consistent behavior across episodes
3. **All Pokemon Types**: Test with various team compositions
4. **Edge Cases**: Empty available_switches, fainted Pokemon, etc.

### Expected Behavior
- Action masks should match `action_helper.get_action_mapping()` directly
- No all-zero action masks (except in terminal game states)
- Switch actions available based on battle state, not team preview

## 📝 Documentation Updates

### Comments Updated
- `get_action_mask()` docstring: Removed team restriction references
- `_compute_all_masks()`: Added comment about removal
- Code structure: Simplified flow without restriction checks

### Files Modified
- `src/env/pokemon_env.py`: Main implementation changes
- `config/train_config.yml`: Minor configuration updates

## 🔄 Migration Notes

### For Developers
- **No API Changes**: External interfaces remain the same
- **Behavior Change**: More permissive action masks
- **Testing**: Verify no regressions in training stability

### For Users
- **Training Benefits**: More stable training sessions
- **Pokemon Support**: Better support for all Pokemon forms
- **No Action Required**: Change is transparent to training scripts

## 🎯 Future Considerations

### Alternative Approaches (Not Implemented)
1. **Form Change Mapping**: Create comprehensive form change database
2. **Species Normalization**: Normalize all forms to base species
3. **Flexible Matching**: Use fuzzy matching for species names

### Why Not Chosen
- **Complexity**: Requires extensive Pokemon knowledge database
- **Maintenance**: Ongoing updates needed for new Pokemon/forms
- **Risk**: Potential for new edge cases and bugs

## 🏁 Conclusion

The team restriction system removal prioritizes training stability and simplicity over strict competitive rule enforcement. This change eliminates a major source of training failures while maintaining the core functionality of the Pokemon battle environment.

The underlying `poke-env` library continues to handle battle validity, ensuring that only legitimate actions are available, just without the additional team preview restrictions that were causing form change issues.