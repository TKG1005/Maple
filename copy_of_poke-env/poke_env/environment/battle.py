from logging import Logger
from typing import Any, Dict, List, Optional, Union

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType


class Battle(AbstractBattle):
    def __init__(
        self,
        battle_tag: str,
        username: str,
        logger: Logger,
        gen: int,
        save_replays: Union[str, bool] = False,
    ):
        super(Battle, self).__init__(battle_tag, username, logger, save_replays, gen)

        # Turn choice attributes
        self._available_moves: List[Move] = []
        self._available_switches: List[Pokemon] = []
        self._can_dynamax: bool = False
        self._can_mega_evolve: bool = False
        self._can_tera: Optional[PokemonType] = None
        self._can_z_move: bool = False
        self._opponent_can_dynamax = True
        self._opponent_can_mega_evolve = True
        self._opponent_can_z_move = True
        self._opponent_can_tera: bool = False
        self._force_switch: bool = False
        self._maybe_trapped: bool = False
        self._trapped: bool = False

    def _log_team_state(self, label: str) -> None:
        """Log each team Pokemon's active and fainted status for debugging."""
        if self.logger is None:
            return
        try:
            info = [
                f"{ident}: active={mon.active}, fainted={mon.fainted}"
                for ident, mon in self.team.items()
            ]
            self.logger.debug(
                "[DBG] %s %s team_state: %s", self.battle_tag, label, info
            )
        except Exception as exc:  # pragma: no cover - debug helper
            self.logger.debug(
                "[DBG] failed to log team_state %s: %s", label, exc
            )

    def clear_all_boosts(self):
        if self.active_pokemon is not None:
            self.active_pokemon.clear_boosts()
        if self.opponent_active_pokemon is not None:
            self.opponent_active_pokemon.clear_boosts()

    def end_illusion(self, pokemon_name: str, details: str):
        if pokemon_name[:2] == self._player_role:
            active = self.active_pokemon
        else:
            active = self.opponent_active_pokemon

        if active is None:
            raise ValueError("Cannot end illusion without an active pokemon.")

        self._end_illusion_on(
            illusioned=active, illusionist=pokemon_name, details=details
        )

    def parse_request(self, request: Dict[str, Any]) -> None:
        """
        Update the object from a request.
        The player's pokemon are all updated, as well as available moves, switches and
        other related information (z move, mega evolution, forced switch...).

        :param request: Parsed JSON request object.
        :type request: dict
        """
        if "wait" in request and request["wait"]:
            self._wait = True
        else:
            self._wait = False

        side = request["side"]

        self._available_moves = []
        self._available_switches = []
        self._can_mega_evolve = False
        self._can_z_move = False
        self._can_dynamax = False
        self._can_tera = None
        self._maybe_trapped = False
        self._reviving = any(
            [m["reviving"] for m in side.get("pokemon", []) if "reviving" in m]
        )
        self._trapped = False
        self._force_switch = request.get("forceSwitch", [False])[0]

        if self._force_switch:
            self._move_on_next_request = True

        self._last_request = request

        if request.get("teamPreview", False):
            self._teampreview = True
            number_of_mons = len(request["side"]["pokemon"])
            self._max_team_size = request.get("maxTeamSize", number_of_mons)
        else:
            self._teampreview = False
        self._update_team_from_request(request["side"])

        if "active" in request:
            active_request = request["active"][0]

            if active_request.get("trapped"):
                self._trapped = True

            if self.active_pokemon is not None:
                self._available_moves.extend(
                    self.active_pokemon.available_moves_from_request(active_request)
                )
            if active_request.get("canMegaEvo", False):
                self._can_mega_evolve = True
            if active_request.get("canZMove", False):
                self._can_z_move = True
            if active_request.get("canDynamax", False):
                self._can_dynamax = True
            if active_request.get("maybeTrapped", False):
                self._maybe_trapped = True
            if active_request.get("canTerastallize", False):
                self._can_tera = PokemonType.from_name(
                    active_request["canTerastallize"]
                )

        if side["pokemon"]:
            self._player_role = side["pokemon"][0]["ident"][:2]

        if not self.trapped and not self.reviving:
            for pokemon in side["pokemon"]:
                if pokemon:
                    pokemon = self._team[pokemon["ident"]]
                    if not pokemon.active and not pokemon.fainted:
                        self._available_switches.append(pokemon)

        if not self.trapped and self.reviving:
            for pokemon in side["pokemon"]:
                if pokemon and pokemon.get("reviving", False):
                    pokemon = self._team[pokemon["ident"]]
                    if not pokemon.active:
                        self._available_switches.append(pokemon)

        self._log_team_state("after parse_request")

    def switch(self, pokemon_str: str, details: str, hp_status: str):
        identifier = pokemon_str.split(":")[0][:2]

        if identifier == self._player_role:
            if self.active_pokemon:
                self.active_pokemon.switch_out()
        else:
            if self.opponent_active_pokemon:
                self.opponent_active_pokemon.switch_out()

        pokemon = self.get_pokemon(pokemon_str, details=details)

        pokemon.switch_in(details=details)
        pokemon.set_hp_status(hp_status)

        self._log_team_state("after switch")

    @property
    def active_pokemon(self) -> Optional[Pokemon]:
        """
        :return: The active pokemon
        :rtype: Optional[Pokemon]
        """
        for pokemon in self.team.values():
            if pokemon.active:
                return pokemon
        return None

    @property
    def all_active_pokemons(self) -> List[Optional[Pokemon]]:
        """
        :return: A list containing all active pokemons and/or Nones.
        :rtype: List[Optional[Pokemon]]
        """
        return [self.active_pokemon, self.opponent_active_pokemon]

    @property
    def available_moves(self) -> List[Move]:
        """
        :return: The list of moves the player can use during the current move request.
        :rtype: List[Move]
        """
        return self._available_moves

    @property
    def available_switches(self) -> List[Pokemon]:
        """
        :return: The list of switches the player can do during the current move request.
        :rtype: List[Pokemon]
        """
        return self._available_switches

    @property
    def can_dynamax(self) -> bool:
        """
        :return: Whether or not the current active pokemon can dynamax
        :rtype: bool
        """
        return self._can_dynamax

    @property
    def can_mega_evolve(self) -> bool:
        """
        :return: Whether or not the current active pokemon can mega evolve.
        :rtype: bool
        """
        return self._can_mega_evolve

    @property
    def can_tera(self) -> Optional[PokemonType]:
        """
        :return: None, or the type the active pokemon can terastallize into.
        :rtype: PokemonType, optional
        """
        return self._can_tera

    @property
    def can_z_move(self) -> bool:
        """
        :return: Whether or not the current active pokemon can z-move.
        :rtype: bool
        """
        return self._can_z_move

    @property
    def force_switch(self) -> bool:
        """
        :return: A boolean indicating whether the active pokemon is forced to switch
            out.
        :rtype: Optional[bool]
        """
        return self._force_switch

    @property
    def grounded(self) -> bool:
        """
        :return: A boolean indicating whether the active pokemon is grounded
        :rtype: bool
        """
        return self.is_grounded(self.active_pokemon) if self.active_pokemon else True

    @property
    def maybe_trapped(self) -> bool:
        """
        :return: A boolean indicating whether the active pokemon is maybe trapped by the
            opponent.
        :rtype: bool
        """
        return self._maybe_trapped

    @property
    def opponent_active_pokemon(self) -> Optional[Pokemon]:
        """
        :return: The opponent active pokemon
        :rtype: Pokemon
        """
        for pokemon in self.opponent_team.values():
            if pokemon.active:
                return pokemon
        return None

    @property
    def opponent_can_dynamax(self) -> bool:
        """
        :return: Whether or not opponent's current active pokemon can dynamax
        :rtype: bool
        """
        return self._opponent_can_dynamax

    @opponent_can_dynamax.setter
    def opponent_can_dynamax(self, value: bool):
        self._opponent_can_dynamax = value

    @property
    def opponent_can_mega_evolve(self) -> Union[bool, List[bool]]:
        """
        :return: Whether or not opponent's current active pokemon can mega-evolve
        :rtype: bool
        """
        return self._opponent_can_mega_evolve

    @opponent_can_mega_evolve.setter
    def opponent_can_mega_evolve(self, value: bool):
        self._opponent_can_mega_evolve = value

    @property
    def opponent_can_tera(self) -> bool:
        """
        :return: Whether or not opponent's current active pokemon can terastallize
        :rtype: bool
        """
        return self._opponent_can_tera

    @property
    def opponent_can_z_move(self) -> Union[bool, List[bool]]:
        """
        :return: Whether or not opponent's current active pokemon can z-move
        :rtype: bool
        """
        return self._opponent_can_z_move

    @opponent_can_z_move.setter
    def opponent_can_z_move(self, value: bool):
        self._opponent_can_z_move = value

    @property
    def trapped(self) -> bool:
        """
        :return: A boolean indicating whether the active pokemon is trapped, either by
            the opponent or as a side effect of one your moves.
        :rtype: bool
        """
        return self._trapped

    @trapped.setter
    def trapped(self, value: bool):
        self._trapped = value
