/Users/takagikouichi/Documents/GitHub/Maple
├── config
│   ├── action_map.yml
│   ├── battle_available_notes.md
│   ├── generate_yaml.py
│   ├── move_catalog.md
│   ├── moves_english_japanese.csv
│   ├── my_team_for_debug.txt
│   ├── my_team.txt
│   ├── pokemon_all_moves.csv
│   ├── state_feature_catalog_temp - シート1.csv
│   └── state_spec.yml
├── docs
│   └── AI-design
│       ├── M2
│       ├── M4
│       └── PokemonEnv_Specification.md
├── maple_structure.md
├── pokemon-showdown
│   ├── ARCHITECTURE.md
│   ├── build
│   ├── CODEOWNERS
│   ├── COMMANDLINE.md
│   ├── config
│   │   ├── avatars
│   │   ├── avatars.json
│   │   ├── chat-plugins
│   │   ├── chatrooms.json
│   │   ├── config-example.js
│   │   ├── config.js
│   │   ├── custom-example.css
│   │   ├── CUSTOM-RULES.md
│   │   ├── formats.ts
│   │   ├── hosts.csv
│   │   ├── ladders
│   │   ├── proxies.csv
│   │   └── suspects.json
│   ├── CONTRIBUTING.md
│   ├── data
│   │   ├── abilities.ts
│   │   ├── aliases.ts
│   │   ├── cg-team-data.ts
│   │   ├── cg-teams.ts
│   │   ├── conditions.ts
│   │   ├── formats-data.ts
│   │   ├── FORMES.md
│   │   ├── items.ts
│   │   ├── learnsets.ts
│   │   ├── mods
│   │   ├── moves.ts
│   │   ├── natures.ts
│   │   ├── pokedex.ts
│   │   ├── pokemongo.ts
│   │   ├── random-battles
│   │   ├── rulesets.ts
│   │   ├── scripts.ts
│   │   ├── tags.ts
│   │   ├── text
│   │   └── typechart.ts
│   ├── databases
│   │   ├── chat-plugins.db
│   │   ├── migrations
│   │   ├── offline-pms.db
│   │   └── schemas
│   ├── dist
│   │   ├── config
│   │   ├── data
│   │   ├── lib
│   │   ├── server
│   │   ├── sim
│   │   ├── tools
│   │   └── translations
│   ├── eslint-ps-standard.mjs
│   ├── eslint.config.mjs
│   ├── lib
│   │   ├── crashlogger.ts
│   │   ├── DASHYCODE.md
│   │   ├── dashycode.ts
│   │   ├── database.ts
│   │   ├── fs.ts
│   │   ├── index.ts
│   │   ├── net.ts
│   │   ├── postgres.ts
│   │   ├── process-manager.ts
│   │   ├── repl.ts
│   │   ├── sql.ts
│   │   ├── STREAMS.md
│   │   ├── streams.ts
│   │   └── utils.ts
│   ├── LICENSE
│   ├── logs
│   │   ├── chat
│   │   ├── chatlog-access.txt
│   │   ├── errors.txt
│   │   ├── lastbattle.txt
│   │   ├── logging.md
│   │   ├── modlog
│   │   ├── randbats
│   │   ├── repl
│   │   ├── responder.jsonl
│   │   └── tickets
│   ├── node_modules
│   │   ├── @esbuild
│   │   ├── @eslint
│   │   ├── @eslint-community
│   │   ├── @gar
│   │   ├── @humanfs
│   │   ├── @humanwhocodes
│   │   ├── @isaacs
│   │   ├── @nodelib
│   │   ├── @npmcli
│   │   ├── @pkgjs
│   │   ├── @stylistic
│   │   ├── @tootallnate
│   │   ├── @types
│   │   ├── @typescript-eslint
│   │   ├── abbrev
│   │   ├── acorn
│   │   ├── acorn-jsx
│   │   ├── agent-base
│   │   ├── agentkeepalive
│   │   ├── aggregate-error
│   │   ├── ajv
│   │   ├── ansi-colors
│   │   ├── ansi-regex
│   │   ├── ansi-styles
│   │   ├── any-promise
│   │   ├── anymatch
│   │   ├── aproba
│   │   ├── are-we-there-yet
│   │   ├── argparse
│   │   ├── balanced-match
│   │   ├── base64-js
│   │   ├── better-sqlite3
│   │   ├── binary-extensions
│   │   ├── bindings
│   │   ├── bl
│   │   ├── brace-expansion
│   │   ├── braces
│   │   ├── browser-stdout
│   │   ├── buffer
│   │   ├── buffer-from
│   │   ├── buffer-writer
│   │   ├── cacache
│   │   ├── callsites
│   │   ├── camelcase
│   │   ├── chalk
│   │   ├── chokidar
│   │   ├── chownr
│   │   ├── clean-stack
│   │   ├── cliui
│   │   ├── cloud-env
│   │   ├── color-convert
│   │   ├── color-name
│   │   ├── color-support
│   │   ├── colors
│   │   ├── commander
│   │   ├── concat-map
│   │   ├── console-control-strings
│   │   ├── cross-spawn
│   │   ├── debug
│   │   ├── decamelize
│   │   ├── decompress-response
│   │   ├── deep-extend
│   │   ├── deep-is
│   │   ├── delegates
│   │   ├── denque
│   │   ├── detect-libc
│   │   ├── diff
│   │   ├── docopt
│   │   ├── eastasianwidth
│   │   ├── emoji-regex
│   │   ├── encoding
│   │   ├── end-of-stream
│   │   ├── env-paths
│   │   ├── err-code
│   │   ├── esbuild
│   │   ├── escalade
│   │   ├── escape-string-regexp
│   │   ├── eslint
│   │   ├── eslint-scope
│   │   ├── eslint-visitor-keys
│   │   ├── espree
│   │   ├── esquery
│   │   ├── esrecurse
│   │   ├── estraverse
│   │   ├── esutils
│   │   ├── expand-template
│   │   ├── fast-deep-equal
│   │   ├── fast-glob
│   │   ├── fast-json-stable-stringify
│   │   ├── fast-levenshtein
│   │   ├── fastq
│   │   ├── faye-websocket
│   │   ├── file-entry-cache
│   │   ├── file-uri-to-path
│   │   ├── fill-range
│   │   ├── find-up
│   │   ├── flat
│   │   ├── flat-cache
│   │   ├── flatted
│   │   ├── foreground-child
│   │   ├── fs-constants
│   │   ├── fs-minipass
│   │   ├── fs.realpath
│   │   ├── fsevents
│   │   ├── gauge
│   │   ├── generate-function
│   │   ├── get-caller-file
│   │   ├── get-stdin
│   │   ├── github-from-package
│   │   ├── githubhook
│   │   ├── glob
│   │   ├── glob-parent
│   │   ├── globals
│   │   ├── graceful-fs
│   │   ├── graphemer
│   │   ├── has-flag
│   │   ├── has-unicode
│   │   ├── he
│   │   ├── http-cache-semantics
│   │   ├── http-parser-js
│   │   ├── http-proxy-agent
│   │   ├── https-proxy-agent
│   │   ├── humanize-ms
│   │   ├── iconv-lite
│   │   ├── ieee754
│   │   ├── ignore
│   │   ├── import-fresh
│   │   ├── imurmurhash
│   │   ├── indent-string
│   │   ├── infer-owner
│   │   ├── inflight
│   │   ├── inherits
│   │   ├── ini
│   │   ├── ip-address
│   │   ├── is-binary-path
│   │   ├── is-extglob
│   │   ├── is-fullwidth-code-point
│   │   ├── is-glob
│   │   ├── is-lambda
│   │   ├── is-number
│   │   ├── is-plain-obj
│   │   ├── is-property
│   │   ├── is-unicode-supported
│   │   ├── isexe
│   │   ├── jackspeak
│   │   ├── js-yaml
│   │   ├── jsbn
│   │   ├── json-buffer
│   │   ├── json-schema-traverse
│   │   ├── json-stable-stringify-without-jsonify
│   │   ├── keyv
│   │   ├── levn
│   │   ├── lines-and-columns
│   │   ├── locate-path
│   │   ├── lodash.merge
│   │   ├── log-symbols
│   │   ├── long
│   │   ├── lru-cache
│   │   ├── make-fetch-happen
│   │   ├── merge2
│   │   ├── micromatch
│   │   ├── mime
│   │   ├── mimic-response
│   │   ├── minimatch
│   │   ├── minimist
│   │   ├── minipass
│   │   ├── minipass-collect
│   │   ├── minipass-fetch
│   │   ├── minipass-flush
│   │   ├── minipass-pipeline
│   │   ├── minipass-sized
│   │   ├── minizlib
│   │   ├── mkdirp
│   │   ├── mkdirp-classic
│   │   ├── mocha
│   │   ├── ms
│   │   ├── multiline
│   │   ├── mysql2
│   │   ├── mz
│   │   ├── named-placeholders
│   │   ├── napi-build-utils
│   │   ├── natural-compare
│   │   ├── needle
│   │   ├── negotiator
│   │   ├── node-abi
│   │   ├── node-gyp
│   │   ├── node-static
│   │   ├── nodemailer
│   │   ├── nopt
│   │   ├── normalize-path
│   │   ├── npmlog
│   │   ├── object-assign
│   │   ├── once
│   │   ├── optimist
│   │   ├── optionator
│   │   ├── p-limit
│   │   ├── p-locate
│   │   ├── p-map
│   │   ├── package-json-from-dist
│   │   ├── packet-reader
│   │   ├── parent-module
│   │   ├── path-exists
│   │   ├── path-is-absolute
│   │   ├── path-key
│   │   ├── path-scurry
│   │   ├── permessage-deflate
│   │   ├── pg
│   │   ├── pg-cloudflare
│   │   ├── pg-connection-string
│   │   ├── pg-int8
│   │   ├── pg-pool
│   │   ├── pg-protocol
│   │   ├── pg-types
│   │   ├── pgpass
│   │   ├── picomatch
│   │   ├── pirates
│   │   ├── postgres-array
│   │   ├── postgres-bytea
│   │   ├── postgres-date
│   │   ├── postgres-interval
│   │   ├── preact
│   │   ├── preact-render-to-string
│   │   ├── prebuild-install
│   │   ├── prelude-ls
│   │   ├── probe-image-size
│   │   ├── promise-inflight
│   │   ├── promise-retry
│   │   ├── pump
│   │   ├── punycode
│   │   ├── queue-microtask
│   │   ├── randombytes
│   │   ├── rc
│   │   ├── readable-stream
│   │   ├── readdirp
│   │   ├── require-directory
│   │   ├── resolve-from
│   │   ├── retry
│   │   ├── reusify
│   │   ├── rimraf
│   │   ├── run-parallel
│   │   ├── safe-buffer
│   │   ├── safer-buffer
│   │   ├── sax
│   │   ├── semver
│   │   ├── seq-queue
│   │   ├── serialize-javascript
│   │   ├── set-blocking
│   │   ├── shebang-command
│   │   ├── shebang-regex
│   │   ├── signal-exit
│   │   ├── simple-concat
│   │   ├── simple-get
│   │   ├── smart-buffer
│   │   ├── smogon
│   │   ├── sockjs
│   │   ├── socks
│   │   ├── socks-proxy-agent
│   │   ├── source-map
│   │   ├── source-map-support
│   │   ├── split2
│   │   ├── sprintf-js
│   │   ├── sql-template-strings
│   │   ├── sqlite
│   │   ├── sqlstring
│   │   ├── ssri
│   │   ├── stream-parser
│   │   ├── string_decoder
│   │   ├── string-width
│   │   ├── string-width-cjs
│   │   ├── strip-ansi
│   │   ├── strip-ansi-cjs
│   │   ├── strip-indent
│   │   ├── strip-json-comments
│   │   ├── sucrase
│   │   ├── supports-color
│   │   ├── tar
│   │   ├── tar-fs
│   │   ├── tar-stream
│   │   ├── thenify
│   │   ├── thenify-all
│   │   ├── to-regex-range
│   │   ├── ts-api-utils
│   │   ├── ts-chacha20
│   │   ├── ts-interface-checker
│   │   ├── tunnel-agent
│   │   ├── type-check
│   │   ├── typescript
│   │   ├── typescript-eslint
│   │   ├── unique-filename
│   │   ├── unique-slug
│   │   ├── uri-js
│   │   ├── util-deprecate
│   │   ├── websocket-driver
│   │   ├── websocket-extensions
│   │   ├── which
│   │   ├── word-wrap
│   │   ├── wordwrap
│   │   ├── workerpool
│   │   ├── wrap-ansi
│   │   ├── wrap-ansi-cjs
│   │   ├── wrappy
│   │   ├── xtend
│   │   ├── y18n
│   │   ├── yallist
│   │   ├── yargs
│   │   ├── yargs-parser
│   │   ├── yargs-unparser
│   │   └── yocto-queue
│   ├── old-simulator-doc.txt
│   ├── package-lock.json
│   ├── package.json
│   ├── pokemon-showdown
│   ├── Procfile
│   ├── PROTOCOL.md
│   ├── README.md
│   ├── server
│   │   ├── artemis
│   │   ├── chat-commands
│   │   ├── chat-formatter.ts
│   │   ├── chat-jsx.tsx
│   │   ├── chat-plugins
│   │   ├── chat.ts
│   │   ├── config-loader.ts
│   │   ├── friends.ts
│   │   ├── global-types.ts
│   │   ├── global-variables.d.ts
│   │   ├── index.ts
│   │   ├── ip-tools.ts
│   │   ├── ladders-challenges.ts
│   │   ├── ladders-local.ts
│   │   ├── ladders-remote.ts
│   │   ├── ladders.ts
│   │   ├── loginserver.ts
│   │   ├── modlog
│   │   ├── monitor.ts
│   │   ├── private-messages
│   │   ├── punishments.ts
│   │   ├── README.md
│   │   ├── replays.ts
│   │   ├── room-battle-bestof.ts
│   │   ├── room-battle.ts
│   │   ├── room-game.ts
│   │   ├── room-minor-activity.ts
│   │   ├── roomlogs.ts
│   │   ├── rooms.ts
│   │   ├── sockets.ts
│   │   ├── static
│   │   ├── team-validator-async.ts
│   │   ├── tournaments
│   │   ├── user-groups.ts
│   │   ├── users.ts
│   │   └── verifier.ts
│   ├── sim
│   │   ├── battle-actions.ts
│   │   ├── battle-queue.ts
│   │   ├── battle-stream.ts
│   │   ├── battle.ts
│   │   ├── dex-abilities.ts
│   │   ├── dex-conditions.ts
│   │   ├── dex-data.ts
│   │   ├── dex-formats.ts
│   │   ├── dex-items.ts
│   │   ├── dex-moves.ts
│   │   ├── dex-species.ts
│   │   ├── DEX.md
│   │   ├── dex.ts
│   │   ├── examples
│   │   ├── field.ts
│   │   ├── global-types.ts
│   │   ├── index.ts
│   │   ├── NONSTANDARD.md
│   │   ├── pokemon.ts
│   │   ├── prng.ts
│   │   ├── README.md
│   │   ├── side.ts
│   │   ├── SIM-PROTOCOL.md
│   │   ├── SIMULATOR.md
│   │   ├── state.ts
│   │   ├── team-validator.ts
│   │   ├── TEAMS.md
│   │   ├── teams.ts
│   │   ├── tools
│   │   └── tsconfig.json
│   ├── simulator-doc.txt
│   ├── test
│   │   ├── assert.js
│   │   ├── common.js
│   │   ├── lib
│   │   ├── main.js
│   │   ├── random-battles
│   │   ├── replays
│   │   ├── server
│   │   ├── sim
│   │   ├── TESTS.md
│   │   ├── tools
│   │   └── users-utils.js
│   ├── tools
│   │   ├── build-utils.js
│   │   ├── modlog
│   │   ├── set-import
│   │   ├── simulate
│   │   └── team-generation-benchmark.js
│   ├── translations
│   │   ├── dutch
│   │   ├── english
│   │   ├── french
│   │   ├── german
│   │   ├── hindi
│   │   ├── italian
│   │   ├── japanese
│   │   ├── portuguese
│   │   ├── README.md
│   │   ├── simplifiedchinese
│   │   ├── spanish
│   │   ├── traditionalchinese
│   │   └── turkish
│   └── tsconfig.json
├── pytest.ini
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-312.pyc
│   ├── action
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── action_helper.py
│   ├── agents
│   │   ├── __pycache__
│   │   ├── my_simple_player.py
│   │   ├── queued_random_player.py
│   │   └── rule_based_player.py
│   ├── env
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── pokemon_env.py
│   ├── run_local_battle.py
│   ├── state
│   │   ├── __pycache__
│   │   ├── generate_yaml.py
│   │   ├── state_feature_catalog_temp - シート1.csv
│   │   └── state_observer.py
│   └── util
└── test
    ├── __pycache__
    │   ├── run_battle.cpython-312.pyc
    │   ├── test_env.cpython-312-pytest-8.4.0.pyc
    │   ├── test_pokemon_env_showdown.cpython-312-pytest-8.4.0.pyc
    │   ├── test_pokemon_env_step2_4.cpython-312-pytest-8.4.0.pyc
    │   ├── test_pokemon_env_step5.cpython-312-pytest-8.4.0.pyc
    │   └── test_run_battle.cpython-312-pytest-8.4.0.pyc
    ├── run_battle.py
    ├── simple_agent.py
    └── test_run_battle.py

390 directories, 166 files
