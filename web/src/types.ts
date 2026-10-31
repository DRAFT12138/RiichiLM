export type Mode = "hanchan" | "tonpuu" | "single";
export interface SetupConfig {
  mode: Mode;
  opponents: string[];
  spectator: boolean;
  primary: string;
}
export type SpectatorControl = "pause" | "resume" | "step" | "view" | "speed";
export interface Controller {
  kind: string;
  checkpoint_id: string | null;
  name?: string | null;
  model?: string | null;
  sampling?: string | null;
}
export interface DecisionNote {
  seat: number;
  round_id: number;
  revision: number;
  action: { kind: string; args: (string | number)[] };
  candidate_count: number;
  summary: string | null;
  summary_source: string | null;
  sampling: string;
  selected_probability: number | null;
  seconds: number;
}
export interface Checkpoint {
  id: string;
  checkpoint_id: string;
  attempt_id: string;
  step: number;
  created_at: string;
  selectable: boolean;
  status: string;
}
export interface Attempt {
  attempt_id: string;
  started_at: string;
  source_checkpoint: {
    checkpoint_id: string;
    run_id: string;
    attempt_id: string;
    global_step: number;
  } | null;
  progress: { global_step?: number } | null;
  outcome: string;
}
export interface Run {
  run_id: string;
  experiment: string;
  created_at: string;
  training_type: string;
  attempts: Attempt[];
  checkpoints: Checkpoint[];
  errors: string[];
}
export interface Catalog {
  runs: Run[];
  errors: string[];
  llms?: { id: string; name: string; model: string }[];
}
export interface Candidate {
  id: string;
  kind: string;
  args: (string | number)[];
}
export interface Meld {
  name: string;
  tiles: string[];
  base_tile: string;
  from?: number | null;
  called_tile?: string | null;
}
export interface Player {
  seat: number;
  wind: string;
  score: number;
  dealer: boolean;
  riichi: boolean;
  hand: string[] | null;
  drawn: string | null;
  hand_count: number;
  melds: Meld[];
  river: {
    tile: string;
    tsumogiri: boolean;
    riichi: boolean;
    called: boolean;
  }[];
}
export interface Winner {
  actor: number;
  target: number;
  tsumo: boolean;
  hand: string[];
  winning_tile: string;
  melds: Meld[];
  ura_indicators: string[];
  han: number | null;
  fu: number | null;
  yakuman_units: number;
  yaku: string[];
  dora: number;
  aka: number;
  ura: number;
}
export interface RoundResult {
  round_id: number;
  revision: number;
  kind: string;
  reason: string | null;
  winners: Winner[];
  score_delta: number[];
  scores: number[];
  kyotaku: number;
  game_over: boolean;
}
export interface Table {
  table_id: string;
  page_id: string;
  view_version: number;
  state: "loading" | "playing" | "round_result" | "finished" | "failed";
  mode: Mode;
  spectator?: boolean;
  viewer?: number;
  paused?: boolean;
  playback_interval?: number;
  decision_notes?: DecisionNote[];
  controllers?: Controller[];
  busy: boolean;
  error?: string;
  command_error?: string;
  round_id: number;
  revision: number;
  round_wind: string;
  round_number: number;
  honba: number;
  kyotaku: number;
  live_remaining: number;
  dora_indicators: string[];
  players?: Player[];
  active_seats: number[];
  response_window: boolean;
  response_trigger: {
    source: number;
    tile: string;
    kind: "discard" | "kan";
  } | null;
  window: Record<string, string | number> | null;
  candidates: Candidate[];
  thinking: number | null;
  result: RoundResult | null;
  final?: { scores: number[]; ranks: number[] };
  cross_mode: boolean;
  timings?: {
    load_seconds: number;
    last_decision_seconds: number | null;
    peak_rss_mib: number;
  };
  opponents?: Controller[];
}
