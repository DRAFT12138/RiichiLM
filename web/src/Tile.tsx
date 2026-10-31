import { useI18n, type Translator } from "./i18n";
const artwork = import.meta.glob<string>("./assets/tiles/*.svg", {
  eager: true,
  query: "?url",
  import: "default",
});
const image = (name: string) => artwork[`./assets/tiles/${name}.svg`];
const honors: Record<string, string> = {
  E: "Ton",
  S: "Nan",
  W: "Shaa",
  N: "Pei",
  P: "Haku",
  F: "Hatsu",
  C: "Chun",
};
export const tileArtwork = (tile: string) => {
  if (honors[tile]) return image(honors[tile]);
  const match = /^([1-9])([mps])(r?)$/.exec(tile);
  if (!match) return image("Blank");
  const [, rank, suit, red] = match;
  const family = { m: "Man", p: "Pin", s: "Sou" }[suit];
  return image(`${family}${rank}${red && rank === "5" ? "-Dora" : ""}`);
};
const tileName = (tile: string | undefined, t: Translator, locale: string) => {
  if (!tile) return undefined;
  const honorNames: Record<string, string> = { E: "东", S: "南", W: "西", N: "北", P: "白", F: "发", C: "中" };
  if (honorNames[tile]) return t(honorNames[tile]);
  const match = /^([1-9])([mps])(r?)$/.exec(tile);
  if (!match) return tile;
  const [, rank, suit, red] = match;
  return t("{red}{rank}{suit}", { red: red ? t("赤") : "", rank: locale === "en" ? rank : "一二三四五六七八九"[Number(rank) - 1], suit: t({ m: "万", p: "筒", s: "索" }[suit]!) });
};
export function Tile({
  tile,
  small = false,
  back = false,
  onClick,
  onPreview,
  disabled = false,
  selected = false,
  dim = false,
}: {
  tile?: string;
  small?: boolean;
  back?: boolean;
  onClick?: () => void;
  onPreview?: (tile: string | null) => void;
  disabled?: boolean;
  selected?: boolean;
  dim?: boolean;
}) {
  const { t, locale } = useI18n();
  const className = [
    "tile",
    small && "small",
    back && "back",
    selected && "selected",
    dim && "dim",
    tile?.endsWith("r") && "aka",
  ]
    .filter(Boolean)
    .join(" ");
  const face = (
    <>
      <img
        className="tile-body"
        src={image(back ? "Back" : "Front")}
        alt=""
        draggable={false}
      />
      {!back && tile && (
        <img
          className="tile-face"
          src={tileArtwork(tile)}
          alt=""
          draggable={false}
        />
      )}
    </>
  );
  const label = back ? t("暗牌") : onClick ? t("打出 {tile}", { tile: tile || "" }) : tile;
  return onClick ? (
    <button
      type="button"
      className={className}
      aria-label={label}
      aria-description={tileName(tile, t, locale)}
      title={tileName(tile, t, locale)}
      disabled={disabled}
      onClick={onClick}
      onPointerEnter={() => tile && onPreview?.(tile)}
      onPointerLeave={() => onPreview?.(null)}
      onFocus={() => tile && onPreview?.(tile)}
      onBlur={() => onPreview?.(null)}
    >
      {face}
    </button>
  ) : (
    <span className={className} aria-label={label} aria-description={back ? undefined : tileName(tile, t, locale)} title={back ? undefined : tileName(tile, t, locale)}>
      {face}
    </span>
  );
}
