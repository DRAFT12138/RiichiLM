/** Cosmetic characters are stable for a table and never affect its strategies. */
export interface Cast {
  host: number;
  seats: number[];
}
export const characterNames = [
  "小樱", "小凛", "小澪", "小桃",
  "绯花", "紫苑", "青岚", "花音", "星乃", "若叶",
  "青羽", "桃香", "小葵",
];

export function randomCast(previous?: Cast, random: () => number = Math.random): Cast {
  const pool = characterNames.map((_, id) => id);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const seats = pool.slice(0, 4);
  // Avoid presenting the same lineup twice in a row, even when the shuffle repeats.
  if (previous && seats.every((seat, i) => seat === previous.seats[i])) {
    seats.push(seats.shift()!);
  }
  const choices = characterNames.map((_, id) => id).filter((id) => id !== previous?.host);
  return { host: choices[Math.floor(random() * choices.length)], seats };
}
