/Users/takagikouichi/Documents/GitHub/Maple
├── checkpoints
│   ├── checkpoint_ep1000.pt
│   ├── checkpoint_ep10000.pt
│   ├── checkpoint_ep10500.pt
│   ├── checkpoint_ep11000.pt
│   ├── checkpoint_ep11500.pt
│   ├── checkpoint_ep12000.pt
│   ├── checkpoint_ep12500.pt
│   ├── checkpoint_ep13000.pt
│   ├── checkpoint_ep13500.pt
│   ├── checkpoint_ep14000.pt
│   ├── checkpoint_ep1500.pt
│   ├── checkpoint_ep2000.pt
│   ├── checkpoint_ep2500.pt
│   ├── checkpoint_ep3000.pt
│   ├── checkpoint_ep3500.pt
│   ├── checkpoint_ep4000.pt
│   ├── checkpoint_ep4500.pt
│   ├── checkpoint_ep500.pt
│   ├── checkpoint_ep5000.pt
│   ├── checkpoint_ep5500.pt
│   ├── checkpoint_ep6000.pt
│   ├── checkpoint_ep6500.pt
│   ├── checkpoint_ep7000.pt
│   ├── checkpoint_ep7500.pt
│   ├── checkpoint_ep8000.pt
│   ├── checkpoint_ep8500.pt
│   ├── checkpoint_ep9000.pt
│   └── checkpoint_ep9500.pt
├── CLAUDE.md
├── compare.png
├── config
│   ├── action_map.yml
│   ├── battle_available_notes.md
│   ├── env_config.yml
│   ├── generate_yaml.py
│   ├── moves_english_japanese.csv
│   ├── moves.csv
│   ├── my_team_for_debug.txt
│   ├── my_team.txt
│   ├── pokemon_all_moves.csv
│   ├── pokemon_stats.csv
│   ├── reward.yaml
│   ├── state_feature_catalog_temp - シート1.csv
│   ├── state_spec.yml
│   ├── teams
│   │   ├── balanced_team.txt
│   │   ├── team1.txt
│   │   └── team2.txt
│   ├── train_config.yml
│   └── type_chart.csv
├── copy_of_poke-env
│   └── poke_env
│       ├── __init__.py
│       ├── __pycache__
│       ├── concurrency.py
│       ├── data
│       ├── environment
│       ├── exceptions.py
│       ├── player
│       ├── ps_client
│       ├── py.typed
│       ├── stats.py
│       └── teambuilder
├── data
│   └── copy_of_poke_env
│       └── poke_env
├── docs
│   ├── AI-design
│   │   ├── algorithm_extension_design.md
│   │   ├── M2
│   │   ├── M4
│   │   ├── M4_setup.md
│   │   ├── M5
│   │   ├── M6
│   │   ├── M6_ setup.md
│   │   ├── M7
│   │   └── PokemonEnv_Specification.md
│   ├── damage_calculation_module_requirements.md
│   ├── damage_calculator_design.md
│   ├── Maple_AI_Development_Strategy.md
│   ├── TODO_M7.md
│   └── train_usage.md
├── evaluate_rl.py
├── logs
│   ├── eval_20250707_085705.log
│   ├── eval_20250707_090958.log
│   ├── run1.json
│   ├── train_20250707_102004.log
│   ├── train_20250707_103000.log
│   └── train_20250707_155602.log
├── maple_structure.md
├── maple_structure.txt
├── model_debug.pt
├── model_debug.py
├── model_ppo.pt
├── model_reinforce.pt
├── model.pt
├── plot_compare.py
├── plot_results.py
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
│   │   ├── @alcalzone
│   │   ├── @colors
│   │   ├── @esbuild
│   │   ├── @eslint
│   │   ├── @eslint-community
│   │   ├── @gar
│   │   ├── @humanfs
│   │   ├── @humanwhocodes
│   │   ├── @inkjs
│   │   ├── @isaacs
│   │   ├── @nodelib
│   │   ├── @npmcli
│   │   ├── @openai
│   │   ├── @pkgjs
│   │   ├── @sindresorhus
│   │   ├── @stylistic
│   │   ├── @tokenizer
│   │   ├── @tootallnate
│   │   ├── @types
│   │   ├── @typescript-eslint
│   │   ├── abbrev
│   │   ├── abort-controller
│   │   ├── accepts
│   │   ├── acorn
│   │   ├── acorn-jsx
│   │   ├── agent-base
│   │   ├── agentkeepalive
│   │   ├── aggregate-error
│   │   ├── ajv
│   │   ├── ansi-colors
│   │   ├── ansi-escapes
│   │   ├── ansi-regex
│   │   ├── ansi-styles
│   │   ├── any-promise
│   │   ├── anymatch
│   │   ├── aproba
│   │   ├── are-we-there-yet
│   │   ├── argparse
│   │   ├── asynckit
│   │   ├── auto-bind
│   │   ├── aws-ssl-profiles
│   │   ├── balanced-match
│   │   ├── base64-js
│   │   ├── better-sqlite3
│   │   ├── binary-extensions
│   │   ├── bindings
│   │   ├── bl
│   │   ├── body-parser
│   │   ├── brace-expansion
│   │   ├── braces
│   │   ├── browser-stdout
│   │   ├── buffer
│   │   ├── buffer-from
│   │   ├── buffer-writer
│   │   ├── bundle-name
│   │   ├── bytes
│   │   ├── cacache
│   │   ├── call-bind-apply-helpers
│   │   ├── call-bound
│   │   ├── callsites
│   │   ├── camelcase
│   │   ├── chalk
│   │   ├── char-regex
│   │   ├── chokidar
│   │   ├── chownr
│   │   ├── clean-stack
│   │   ├── cli-boxes
│   │   ├── cli-cursor
│   │   ├── cli-highlight
│   │   ├── cli-spinners
│   │   ├── cli-table3
│   │   ├── cli-truncate
│   │   ├── cliui
│   │   ├── cloud-env
│   │   ├── code-excerpt
│   │   ├── color-convert
│   │   ├── color-name
│   │   ├── color-support
│   │   ├── colors
│   │   ├── combined-stream
│   │   ├── commander
│   │   ├── concat-map
│   │   ├── console-control-strings
│   │   ├── content-disposition
│   │   ├── content-type
│   │   ├── convert-to-spaces
│   │   ├── cookie
│   │   ├── cookie-signature
│   │   ├── cross-spawn
│   │   ├── debug
│   │   ├── decamelize
│   │   ├── decompress-response
│   │   ├── deep-extend
│   │   ├── deep-is
│   │   ├── deepmerge
│   │   ├── default-browser
│   │   ├── default-browser-id
│   │   ├── define-lazy-prop
│   │   ├── delayed-stream
│   │   ├── delegates
│   │   ├── denque
│   │   ├── depd
│   │   ├── detect-libc
│   │   ├── diff
│   │   ├── docopt
│   │   ├── dotenv
│   │   ├── dunder-proto
│   │   ├── eastasianwidth
│   │   ├── ee-first
│   │   ├── emoji-regex
│   │   ├── emojilib
│   │   ├── encodeurl
│   │   ├── encoding
│   │   ├── end-of-stream
│   │   ├── env-paths
│   │   ├── environment
│   │   ├── err-code
│   │   ├── es-define-property
│   │   ├── es-errors
│   │   ├── es-object-atoms
│   │   ├── es-set-tostringtag
│   │   ├── es-toolkit
│   │   ├── esbuild
│   │   ├── escalade
│   │   ├── escape-html
│   │   ├── escape-string-regexp
│   │   ├── eslint
│   │   ├── eslint-scope
│   │   ├── eslint-visitor-keys
│   │   ├── espree
│   │   ├── esquery
│   │   ├── esrecurse
│   │   ├── estraverse
│   │   ├── esutils
│   │   ├── etag
│   │   ├── event-target-shim
│   │   ├── expand-template
│   │   ├── express
│   │   ├── fast-deep-equal
│   │   ├── fast-glob
│   │   ├── fast-json-stable-stringify
│   │   ├── fast-levenshtein
│   │   ├── fast-npm-meta
│   │   ├── fastq
│   │   ├── faye-websocket
│   │   ├── fflate
│   │   ├── figures
│   │   ├── file-entry-cache
│   │   ├── file-type
│   │   ├── file-uri-to-path
│   │   ├── fill-range
│   │   ├── finalhandler
│   │   ├── find-up
│   │   ├── flat
│   │   ├── flat-cache
│   │   ├── flatted
│   │   ├── foreground-child
│   │   ├── form-data
│   │   ├── form-data-encoder
│   │   ├── formdata-node
│   │   ├── forwarded
│   │   ├── fresh
│   │   ├── fs-constants
│   │   ├── fs-minipass
│   │   ├── fs.realpath
│   │   ├── fsevents
│   │   ├── function-bind
│   │   ├── gauge
│   │   ├── generate-function
│   │   ├── get-caller-file
│   │   ├── get-east-asian-width
│   │   ├── get-intrinsic
│   │   ├── get-proto
│   │   ├── get-stdin
│   │   ├── github-from-package
│   │   ├── githubhook
│   │   ├── glob
│   │   ├── glob-parent
│   │   ├── globals
│   │   ├── gopd
│   │   ├── graceful-fs
│   │   ├── graphemer
│   │   ├── has-flag
│   │   ├── has-symbols
│   │   ├── has-tostringtag
│   │   ├── has-unicode
│   │   ├── hasown
│   │   ├── he
│   │   ├── highlight.js
│   │   ├── http-cache-semantics
│   │   ├── http-errors
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
│   │   ├── ink
│   │   ├── ip-address
│   │   ├── ipaddr.js
│   │   ├── is-binary-path
│   │   ├── is-docker
│   │   ├── is-extglob
│   │   ├── is-fullwidth-code-point
│   │   ├── is-glob
│   │   ├── is-in-ci
│   │   ├── is-inside-container
│   │   ├── is-lambda
│   │   ├── is-number
│   │   ├── is-plain-obj
│   │   ├── is-promise
│   │   ├── is-property
│   │   ├── is-unicode-supported
│   │   ├── is-wsl
│   │   ├── isexe
│   │   ├── jackspeak
│   │   ├── js-tokens
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
│   │   ├── loose-envify
│   │   ├── lru-cache
│   │   ├── lru.min
│   │   ├── make-fetch-happen
│   │   ├── marked
│   │   ├── marked-terminal
│   │   ├── math-intrinsics
│   │   ├── media-typer
│   │   ├── meow
│   │   ├── merge-descriptors
│   │   ├── merge2
│   │   ├── micromatch
│   │   ├── mime
│   │   ├── mime-db
│   │   ├── mime-types
│   │   ├── mimic-fn
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
│   │   ├── node-domexception
│   │   ├── node-emoji
│   │   ├── node-fetch
│   │   ├── node-gyp
│   │   ├── node-static
│   │   ├── nodemailer
│   │   ├── nopt
│   │   ├── normalize-path
│   │   ├── npmlog
│   │   ├── object-assign
│   │   ├── object-inspect
│   │   ├── on-finished
│   │   ├── once
│   │   ├── onetime
│   │   ├── open
│   │   ├── openai
│   │   ├── optimist
│   │   ├── optionator
│   │   ├── p-limit
│   │   ├── p-locate
│   │   ├── p-map
│   │   ├── package-json-from-dist
│   │   ├── package-manager-detector
│   │   ├── packet-reader
│   │   ├── parent-module
│   │   ├── parse5
│   │   ├── parse5-htmlparser2-tree-adapter
│   │   ├── parseurl
│   │   ├── patch-console
│   │   ├── path-exists
│   │   ├── path-is-absolute
│   │   ├── path-key
│   │   ├── path-scurry
│   │   ├── path-to-regexp
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
│   │   ├── proxy-addr
│   │   ├── pump
│   │   ├── punycode
│   │   ├── qs
│   │   ├── queue-microtask
│   │   ├── randombytes
│   │   ├── range-parser
│   │   ├── raw-body
│   │   ├── rc
│   │   ├── react
│   │   ├── react-reconciler
│   │   ├── readable-stream
│   │   ├── readdirp
│   │   ├── require-directory
│   │   ├── resolve-from
│   │   ├── restore-cursor
│   │   ├── retry
│   │   ├── reusify
│   │   ├── rimraf
│   │   ├── router
│   │   ├── run-applescript
│   │   ├── run-parallel
│   │   ├── safe-buffer
│   │   ├── safer-buffer
│   │   ├── sax
│   │   ├── scheduler
│   │   ├── semver
│   │   ├── send
│   │   ├── seq-queue
│   │   ├── serialize-javascript
│   │   ├── serve-static
│   │   ├── set-blocking
│   │   ├── setprototypeof
│   │   ├── shebang-command
│   │   ├── shebang-regex
│   │   ├── shell-quote
│   │   ├── side-channel
│   │   ├── side-channel-list
│   │   ├── side-channel-map
│   │   ├── side-channel-weakmap
│   │   ├── signal-exit
│   │   ├── simple-concat
│   │   ├── simple-get
│   │   ├── skin-tone
│   │   ├── slice-ansi
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
│   │   ├── stack-utils
│   │   ├── statuses
│   │   ├── stream-parser
│   │   ├── string_decoder
│   │   ├── string-width
│   │   ├── string-width-cjs
│   │   ├── strip-ansi
│   │   ├── strip-ansi-cjs
│   │   ├── strip-indent
│   │   ├── strip-json-comments
│   │   ├── strtok3
│   │   ├── sucrase
│   │   ├── supports-color
│   │   ├── supports-hyperlinks
│   │   ├── tar
│   │   ├── tar-fs
│   │   ├── tar-stream
│   │   ├── thenify
│   │   ├── thenify-all
│   │   ├── to-regex-range
│   │   ├── to-rotated
│   │   ├── toidentifier
│   │   ├── token-types
│   │   ├── tr46
│   │   ├── ts-api-utils
│   │   ├── ts-chacha20
│   │   ├── ts-interface-checker
│   │   ├── tunnel-agent
│   │   ├── type-check
│   │   ├── type-fest
│   │   ├── type-is
│   │   ├── typescript
│   │   ├── typescript-eslint
│   │   ├── uint8array-extras
│   │   ├── undici-types
│   │   ├── unicode-emoji-modifier-base
│   │   ├── unique-filename
│   │   ├── unique-slug
│   │   ├── unpipe
│   │   ├── uri-js
│   │   ├── use-interval
│   │   ├── util-deprecate
│   │   ├── vary
│   │   ├── web-streams-polyfill
│   │   ├── webidl-conversions
│   │   ├── websocket-driver
│   │   ├── websocket-extensions
│   │   ├── whatwg-url
│   │   ├── which
│   │   ├── widest-line
│   │   ├── word-wrap
│   │   ├── wordwrap
│   │   ├── workerpool
│   │   ├── wrap-ansi
│   │   ├── wrap-ansi-cjs
│   │   ├── wrappy
│   │   ├── ws
│   │   ├── xtend
│   │   ├── y18n
│   │   ├── yallist
│   │   ├── yargs
│   │   ├── yargs-parser
│   │   ├── yargs-unparser
│   │   ├── yocto-queue
│   │   ├── yoga-layout
│   │   └── zod
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
├── README.md
├── replays
│   ├── checkpoint_9775261 - battle-gen9bssregi-262831.html
│   ├── checkpoint_9814917 - battle-gen9bssregi-262832.html
│   ├── checkpoint_9846776 - battle-gen9bssregi-262833.html
│   ├── RuleBasedP_9775261 - battle-gen9bssregi-262831.html
│   ├── RuleBasedP_9814917 - battle-gen9bssregi-262832.html
│   └── RuleBasedP_9846776 - battle-gen9bssregi-262833.html
├── requirements.txt
├── runs
│   └── Jul01_22-47-20_DESKTOP-TDAQCBO
│       └── events.out.tfevents.1751377640.DESKTOP-TDAQCBO.25828.0
├── scripts
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-312.pyc
│   ├── action
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── action_helper.py
│   ├── agents
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── maple_agent_player.py
│   │   ├── MapleAgent.py
│   │   ├── my_simple_player.py
│   │   ├── policy_network.py
│   │   ├── random_agent.py
│   │   ├── replay_buffer.py
│   │   ├── RLAgent.py
│   │   ├── rule_based_player.py
│   │   └── value_network.py
│   ├── algorithms
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── base.py
│   │   ├── dummy.py
│   │   ├── gae.py
│   │   ├── ppo.py
│   │   └── reinforce.py
│   ├── bots
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── max_damage_bot.py
│   │   └── random_bot.py
│   ├── damage
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   └── data_loader.py
│   ├── env
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── env_player.py
│   │   ├── pokemon_env.py
│   │   └── wrappers.py
│   ├── rewards
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── composite.py
│   │   ├── fail_and_immune.py
│   │   ├── hp_delta.py
│   │   ├── knockout.py
│   │   ├── pokemon_count.py
│   │   ├── switch_penalty.py
│   │   └── turn_penalty.py
│   ├── state
│   │   ├── __pycache__
│   │   ├── generate_yaml.py
│   │   ├── state_feature_catalog_temp - シート1.csv
│   │   ├── state_observer.py
│   │   └── type_matchup_extractor.py
│   ├── teams
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── team_loader.py
│   ├── train
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── opponent_pool.py
│   └── util
├── test
│   ├── __init__.py
│   ├── damage
│   │   ├── __init__.py
│   │   └── test_calculator.py
│   ├── run_battle.py
│   └── simple_agent.py
├── test_checkpoints
│   ├── checkpoint_ep14001.pt
│   └── checkpoint_ep14002.pt
├── test_resume.pt
├── tests
│   ├── __pycache__
│   │   ├── test_composite_reward.cpython-312-pytest-8.4.0.pyc
│   │   ├── test_fail_and_immune_reward.cpython-312-pytest-8.4.0.pyc
│   │   └── test_pokemon_count_reward.cpython-312-pytest-8.4.0.pyc
│   ├── test_damage_calculator.py
│   └── test_pokemon_count_reward.py
├── train_rl.py
└── train_selfplay.py

572 directories, 279 files
