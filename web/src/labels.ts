import { useMemo } from "react";
import { useI18n } from "./i18n";
export const modes = {
  hanchan: {
    name: "半庄战",
    tag: "完整体验",
    desc: "东场与南场，沉浸式体验完整对局。",
  },
  tonpuu: {
    name: "东风战",
    tag: "轻松一场",
    desc: "从东一局开始，更轻快的对局节奏。",
  },
  single: {
    name: "单局练习",
    tag: "即开即练",
    desc: "一局定格，专注每一次选择。",
  },
};
export const winds: Record<string, string> = {
  E: "东",
  S: "南",
  W: "西",
  N: "北",
};
export const names = ["你", "下家", "对家", "上家"];
export const observerNames = ["东席", "南席", "西席", "北席"];
export const actionNames: Record<string, string> = {
  A_DISCARD: "打出",
  A_RIICHI_DISCARD: "立直",
  A_CHI: "吃",
  A_PON: "碰",
  A_DAIMINKAN: "明杠",
  A_ANKAN: "暗杠",
  A_KAKAN: "加杠",
  A_RON: "荣和",
  A_TSUMO_WIN: "自摸",
  A_ABORT_KYUUSHU: "九种九牌",
  A_PASS: "跳过",
};
export const yakuNames: Record<string, string> = {
  Y_MENZEN_TSUMO: "门前清自摸和",
  Y_RIICHI: "立直",
  Y_DOUBLE_RIICHI: "两立直",
  Y_IPPATSU: "一发",
  Y_PINFU: "平和",
  Y_TANYAO: "断幺九",
  Y_IIPEIKOU: "一杯口",
  Y_RYANPEIKOU: "二杯口",
  Y_CHANTA: "混全带幺九",
  Y_JUNCHAN: "纯全带幺九",
  Y_ITTSUU: "一气通贯",
  Y_SANSHOKU_DOUJUN: "三色同顺",
  Y_SANSHOKU_DOUKOU: "三色同刻",
  Y_TOITOI: "对对和",
  Y_SANANKOU: "三暗刻",
  Y_SANKANTSU: "三杠子",
  Y_SHOUSANGEN: "小三元",
  Y_HONROUTOU: "混老头",
  Y_CHIITOITSU: "七对子",
  Y_HONITSU: "混一色",
  Y_CHINITSU: "清一色",
  Y_HAKU: "白",
  Y_HATSU: "发",
  Y_CHUN: "中",
  Y_SEAT_WIND: "自风",
  Y_ROUND_WIND: "场风",
  Y_RINSHAN: "岭上开花",
  Y_CHANKAN: "抢杠",
  Y_HAITEI: "海底摸月",
  Y_HOUTEI: "河底捞鱼",
  Y_TENHOU: "天和",
  Y_CHIIHOU: "地和",
  Y_DAISANGEN: "大三元",
  Y_SUUANKOU: "四暗刻",
  Y_SUUANKOU_TANKI: "四暗刻单骑",
  Y_TSUUIISOU: "字一色",
  Y_RYUUIISOU: "绿一色",
  Y_CHINROUTOU: "清老头",
  Y_KOKUSHI: "国士无双",
  Y_KOKUSHI_13: "国士无双十三面",
  Y_SHOUSUUSHII: "小四喜",
  Y_DAISUUSHII: "大四喜",
  Y_SUUKANTSU: "四杠子",
  Y_CHUUREN: "九莲宝灯",
  Y_JUNSEI_CHUUREN: "纯正九莲宝灯",
};

/** Localize display labels while keeping wire identifiers and rules unchanged. */
export function useLabels() {
  const { t } = useI18n();
  return useMemo(() => {
    const record = (source: Record<string, string>) => Object.fromEntries(
      Object.entries(source).map(([key, value]) => [key, t(value)]),
    );
    return {
      modes: Object.fromEntries(Object.entries(modes).map(([key, value]) => [key, {
        name: t(value.name), tag: t(value.tag), desc: t(value.desc),
      }])) as typeof modes,
      winds: record(winds),
      names: names.map((name) => t(name)),
      observerNames: observerNames.map((name) => t(name)),
      actionNames: record(actionNames),
      yakuNames: record(yakuNames),
    };
  }, [t]);
}
