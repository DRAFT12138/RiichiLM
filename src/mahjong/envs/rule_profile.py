"""Complete, explicit profile contract and local backend capability inspection.

Profile validity, source identity, and verified backend support are different
facts. No platform preset or field name constitutes historical admission.
"""
from decimal import Decimal, InvalidOperation

from .offline_mjai import ReplayError, check

YAKUMAN = ('Y_TENHOU','Y_CHIIHOU','Y_DAISANGEN','Y_SUUANKOU','Y_TSUUIISOU','Y_RYUUIISOU',
           'Y_CHINROUTOU','Y_KOKUSHI','Y_SHOUSUUSHII','Y_DAISUUSHII','Y_SUUKANTSU',
           'Y_CHUUREN','Y_SUUANKOU_TANKI','Y_KOKUSHI_13','Y_JUNSEI_CHUUREN')
END_PRIORITY = ('END_BUST','END_DEALER_STOP','END_TARGET_REACHED','END_MAX_ROUND','END_NORMAL')
UNKNOWN = {'$':'UNKNOWN'}


def object_fields(value, names, label):
    check(type(value) is dict,'TYPE_MISMATCH',label)
    expected=set(names.split())
    check(expected<=value.keys(),'MISSING_FIELD',label+': '+','.join(sorted(expected-value.keys())))
    check(value.keys()==expected,'UNKNOWN_FIELD',label+': '+','.join(sorted(value.keys()-expected)))


def integer(value,label,*,minimum=None):
    check(type(value) is int and (minimum is None or value>=minimum),'TYPE_MISMATCH',label)


def boolean(value,label):
    check(type(value) is bool,'TYPE_MISMATCH',label)


def choice(value, values, label):
    check(type(value) in (str,int) and type(value) is not bool and value in values,'TYPE_MISMATCH',label)


def text(value,label):
    check(type(value) is str and bool(value.strip()),'TYPE_MISMATCH',label)


def decimal_json(value,label):
    object_fields(value,'decimal',label)
    text(value['decimal'],label)
    try:
        number=Decimal(value['decimal'])
    except InvalidOperation as error:
        raise ReplayError('TYPE_MISMATCH',label) from error
    check(number.is_finite(),'TYPE_MISMATCH',label)


def rule_options(value):
    object_fields(value,'aka_counts kuitan atozuke multiple_ron kuicae ankan_chankan '
        'riichi_min_score riichi_min_live riichi_kan_policy kazoe kiriage yakuman_multipliers','RuleOptions')
    check(type(value['aka_counts']) is list and len(value['aka_counts'])==3,'TYPE_MISMATCH','aka_counts')
    for v in value['aka_counts']:
        integer(v,'aka_counts',minimum=0)
    for name in ('kuitan','atozuke','kiriage'):
        boolean(value[name],name)
    for name in ('riichi_min_score','riichi_min_live'):
        integer(value[name],name,minimum=0)
    for name,values in {
        'multiple_ron':('MULTIRON_ALL','MULTIRON_TRIPLE_ABORT','MULTIRON_HEAD_BUMP'),
        'kuicae':('KUICAE_ALLOW','KUICAE_SAME','KUICAE_SAME_AND_SUJI'),
        'ankan_chankan':('CHANKAN_NEVER','CHANKAN_KOKUSHI','CHANKAN_GENERAL'),
        'riichi_kan_policy':('RIICHI_KAN_WAITS','RIICHI_KAN_DECOMPOSITION'),
        'kazoe':('KAZOE_YAKUMAN','KAZOE_SANBAIMAN','KAZOE_NONE')}.items():
        choice(value[name],values,name)
    multipliers=value['yakuman_multipliers']
    check(type(multipliers) is list and len(multipliers)==len(YAKUMAN),'TYPE_MISMATCH','yakuman_multipliers')
    for multiplier in multipliers:
        object_fields(multiplier,'name count','Multiplier')
        choice(multiplier['name'],YAKUMAN,'multiplier name')
        integer(multiplier['count'],'multiplier count',minimum=1)
    check([m['name'] for m in multipliers]==list(YAKUMAN),'NONCANONICAL','multiplier identity/order')


def validate_profile(profile):
    object_fields(profile,'ruleset_id revision platform mode rule_options initial_scores return_points '
        'target_points round_policy riichi_policy furiten_policy kan_policy dora_timing score_policy '
        'payment_policy abort_policy report_options','RuleProfile')
    for name in ('ruleset_id','revision'):
        text(profile[name],name)
    choice(profile['platform'],('PLATFORM_TENHOU','PLATFORM_MAJSOUL','PLATFORM_MJAI','PLATFORM_CUSTOM'),'platform')
    choice(profile['mode'],('MODE_HANCHAN','MODE_TONPUU'),'mode')
    rule_options(profile['rule_options'])
    check(type(profile['initial_scores']) is list and len(profile['initial_scores'])==4,'TYPE_MISMATCH','initial_scores')
    for score in profile['initial_scores']:
        integer(score,'initial_scores')
    for name in ('return_points','target_points'):
        integer(profile[name],name,minimum=0)
    value=profile['round_policy']
    object_fields(value,'scheduled_rounds max_rounds bust_below_zero stop_at_target dealer_win_repeat '
        'dealer_tenpai_repeat dealer_win_stop dealer_tenpai_stop tie_order end_priority','round_policy')
    choice(value['scheduled_rounds'],(4,8),'scheduled_rounds')
    choice(value['max_rounds'],(4,8,12,16),'max_rounds')
    check(value['scheduled_rounds']==(8 if profile['mode']=='MODE_HANCHAN' else 4) and
          value['max_rounds']>=value['scheduled_rounds'],'TYPE_MISMATCH','mode/round bounds')
    for name in ('bust_below_zero','stop_at_target','dealer_win_repeat','dealer_tenpai_repeat','dealer_win_stop','dealer_tenpai_stop'):
        boolean(value[name],'round_policy.'+name)
    choice(value['tie_order'],('starting_dealer_rotation',),'tie_order')
    check(type(value['end_priority']) is list and len(value['end_priority'])==5 and
          all(type(v) is str for v in value['end_priority']) and set(value['end_priority'])==set(END_PRIORITY),
          'TYPE_MISMATCH','end_priority complete permutation')
    value=profile['riichi_policy']
    object_fields(value,'deduct_at allow_furiten allow_open ippatsu_cancel_on_call ippatsu_cancel_on_ankan','riichi_policy')
    choice(value['deduct_at'],('accept',),'deduct_at')
    for name in ('allow_furiten','allow_open','ippatsu_cancel_on_call','ippatsu_cancel_on_ankan'):
        boolean(value[name],'riichi_policy.'+name)
    check(not value['allow_open'],'UNSUPPORTED_PROFILE','strict open riichi')
    value=profile['furiten_policy']
    object_fields(value,'miss_ron_on_call temp_clear riichi_clear response_collection','furiten_policy')
    boolean(value['miss_ron_on_call'],'miss_ron_on_call')
    choice(value['temp_clear'],('own_draw',),'temp_clear')
    choice(value['riichi_clear'],('round_start',),'riichi_clear')
    choice(value['response_collection'],('all_before_resolution','priority_short_circuit'),'response_collection')
    value=profile['kan_policy']
    object_fields(value,'max_kan live_cost_at rinshan_cost_at allow_kakan_chankan ankan_public four_kan_abort','kan_policy')
    choice(value['max_kan'],(4,),'max_kan')
    choice(value['live_cost_at'],('rinshan_draw','commit'),'live_cost_at')
    choice(value['rinshan_cost_at'],('draw',),'rinshan_cost_at')
    boolean(value['allow_kakan_chankan'],'allow_kakan_chankan')
    choice(value['ankan_public'],('all_tiles','base_only'),'ankan_public')
    choice(value['four_kan_abort'],('after_discard_response_if_multiple_owners','immediate_if_multiple_owners'),'four_kan_abort')
    value=profile['dora_timing']
    object_fields(value,'ankan daiminkan kakan reveal_before_ron','dora_timing')
    for name in ('ankan','daiminkan','kakan'):
        choice(value[name],('commit','after_discard_response','before_discard_or_next_rinshan'),'dora_timing.'+name)
    boolean(value['reveal_before_ron'],'reveal_before_ron')
    value=profile['score_policy']
    object_fields(value,'min_yaku_han double_wind_pair_fu pinfu_tsumo_fu chiitoitsu_fu yakuman_stack dealer_first_discard','score_policy')
    integer(value['min_yaku_han'],'min_yaku_han',minimum=1)
    for name,values in {'double_wind_pair_fu':(2,4),'pinfu_tsumo_fu':(20,),'chiitoitsu_fu':(25,),
                        'dealer_first_discard':('source','tedashi')}.items():
        choice(value[name],values,'score_policy.'+name)
    boolean(value['yakuman_stack'],'yakuman_stack')
    value=profile['payment_policy']
    object_fields(value,'honba_ron_each honba_tsumo_each multi_ron_honba multi_ron_kyotaku pao_scope '
        'pao_ron pao_tsumo noten_total nagashi nagashi_noten final_kyotaku','payment_policy')
    for name in ('honba_ron_each','honba_tsumo_each','noten_total'):
        integer(value[name],name,minimum=0)
    for name,values in {'multi_ron_honba':('each','nearest'),'multi_ron_kyotaku':('nearest',),
        'pao_scope':('liable_units','all_yakuman'),'pao_ron':('half_with_discarder',),
        'pao_tsumo':('liable_pays_all',),'nagashi':('independent_tsumo','disabled'),
        'final_kyotaku':('first_rank','carry')}.items():
        choice(value[name],values,name)
    boolean(value['nagashi_noten'],'nagashi_noten')
    value=profile['abort_policy']
    object_fields(value,'kyuushu suufon suucha_riichi suukaikan priority','abort_policy')
    for name in ('kyuushu','suufon','suucha_riichi','suukaikan'):
        boolean(value[name],name)
    check(type(value['priority']) is list and len(value['priority'])==5 and
          all(type(v) is str for v in value['priority']) and
          set(value['priority'])=={'ron','four_kan','four_riichi','four_winds','exhaustive'},
          'TYPE_MISMATCH','abort priority')
    value=profile['report_options']
    object_fields(value,'uma oka rounding','report_options')
    if value['uma']!=UNKNOWN:
        check(type(value['uma']) is list and len(value['uma'])==4,'TYPE_MISMATCH','uma')
        for number in value['uma']:
            decimal_json(number,'uma')
    if value['oka']!=UNKNOWN:
        decimal_json(value['oka'],'oka')
    if value['rounding']!=UNKNOWN:
        choice(value['rounding'],('nearest_integer','toward_zero','exact_decimal'),'rounding')
    return profile


def native_rule_kwargs(profile):
    """Explicit mapping only. Unsupported values cannot use a default preset."""
    validate_profile(profile)
    options=profile['rule_options']
    check(options['ankan_chankan']!='CHANKAN_GENERAL','UNSUPPORTED_PROFILE','general ankan chankan')
    check(options['multiple_ron']!='MULTIRON_HEAD_BUMP','UNSUPPORTED_PROFILE','head-bump')
    check(options['kuicae']!='KUICAE_SAME','UNSUPPORTED_PROFILE','same-only kuikae')
    units={m['name']:m['count'] for m in options['yakuman_multipliers']}
    special={'Y_KOKUSHI_13':'is_kokushi_musou_13machi_double',
        'Y_SUUANKOU_TANKI':'is_suuankou_tanki_double','Y_JUNSEI_CHUUREN':'is_junsei_chuurenpoutou_double',
        'Y_DAISUUSHII':'is_daisuushii_double'}
    for name,count in units.items():
        check(count in (1,2) if name in special else count==1,'UNSUPPORTED_PROFILE','multiplier '+name)
    result={symbol:units[name]==2 for name,symbol in special.items()}
    result.update(allows_ron_on_ankan_for_kokushi_musou=options['ankan_chankan']=='CHANKAN_KOKUSHI',
        sanchaho_is_draw=options['multiple_ron']=='MULTIRON_TRIPLE_ABORT',
        kuikae_forbidden=options['kuicae']=='KUICAE_SAME_AND_SUJI',
        yakuman_pao_is_liability_only=profile['payment_policy']['pao_scope']=='liable_units',
        dealer_first_discard_is_tedashi=profile['score_policy']['dealer_first_discard']=='tedashi')
    return result


def hand_rules_from_profile(profile):
    """Compile only the hand-check component; this is not profile admission."""
    from .hand_rules import HandRules
    validate_profile(profile)
    options=profile['rule_options']
    result={name:options[name] for name in ('riichi_min_score','riichi_min_live','riichi_kan_policy')}
    HandRules.from_dict(result)
    return result


def inventory_rules_from_profile(profile):
    """Compile explicit observed-stock capacities, without profile admission."""
    from .tile_stock import TileStock
    validate_profile(profile)
    counts=list(profile['rule_options']['aka_counts'])
    TileStock(counts)
    return dict(aka_counts=counts)


def payment_rules_from_profile(profile):
    """Compile payment facts only; neither win legality nor terminal predicates."""
    from copy import deepcopy
    from .payment_rules import PaymentRules
    validate_profile(profile)
    result=dict(policy=deepcopy(profile['payment_policy']),
                **{name:deepcopy(profile['rule_options'][name]) for name in
                   ('multiple_ron','kazoe','kiriage','yakuman_multipliers')},
                yakuman_stack=profile['score_policy']['yakuman_stack'])
    PaymentRules(result)
    return result


def dora_rules_from_profile(profile):
    """Compile explicit reveal boundaries only, without historical admission."""
    validate_profile(profile)
    timing=profile['dora_timing']
    return dict(dora_timing={name:timing[name] for name in ('ankan','kakan','daiminkan')},
                reveal_before_ron=timing['reveal_before_ron'])


def furiten_rules_from_profile(profile):
    """Compile response knowledge only; source collection evidence is separate."""
    from .furiten import FuritenRules
    validate_profile(profile)
    result=dict(profile['furiten_policy'])
    FuritenRules.from_dict(result)
    return result


def win_rules_from_profile(profile):
    from .win_rules import WinRules
    validate_profile(profile)
    result={name:profile['rule_options'][name] for name in ('kuitan','atozuke')}
    result.update({name:profile['score_policy'][name] for name in
        ('min_yaku_han','double_wind_pair_fu','pinfu_tsumo_fu','chiitoitsu_fu')})
    WinRules.from_dict(result)
    return result


def riichi_policy_from_profile(profile):
    from .riichi_policy import RiichiPolicy
    validate_profile(profile)
    result=dict(profile['riichi_policy'])
    RiichiPolicy.from_dict(result)
    return result


def abort_rules_from_profile(profile):
    from copy import deepcopy
    from .abort_rules import AbortRules
    validate_profile(profile)
    result=dict(deepcopy(profile['abort_policy']),four_kan_abort=profile['kan_policy']['four_kan_abort'])
    AbortRules.from_dict(result)
    return result


def offline_options_from_profile(profile, source_adapter):
    """Assemble the supported configuration; admission remains source-owned.

    Round boundaries use the finite round-termination/1 table. Historical
    admission remains separate; reporting points are not generated here.
    """
    from .offline_mjai import OfflineOptions
    validate_profile(profile)
    revision=source_adapter['revision']
    convlog = revision in ('convlog-mjai/1', 'convlog-mjai/assumed-metadata-1', 'majsoul-mixed-mjai/1')
    check(revision in ('mjai/recorded-deltas-1','gimite-mjai/result-1') or convlog,
          'UNSUPPORTED_PROFILE','source adapter revision: '+revision)
    detailed=revision=='gimite-mjai/result-1'
    check(not convlog or source_adapter['settlement_deltas']=='per_event',
          'UNSUPPORTED_PROFILE','convlog serializes per-winner deltas')
    kan=profile['kan_policy']
    # These resource rules have one implemented behavior; do not use a preset
    # for a configuration that requests another deduction boundary.
    check(kan['max_kan']==4 and kan['rinshan_cost_at']=='draw',
          'UNSUPPORTED_PROFILE','kan resource limits/boundary')
    return OfflineOptions(mode=profile['mode'],ankan_public=kan['ankan_public'],
        allow_ankan_chankan=profile['rule_options']['ankan_chankan']=='CHANKAN_KOKUSHI',
        allow_kakan_chankan=kan['allow_kakan_chankan'],
        dora_timing={k:profile['dora_timing'][k] for k in ('ankan','daiminkan','kakan')},
        reveal_before_ron=profile['dora_timing']['reveal_before_ron'],
        live_cost_at=kan['live_cost_at'],settlement_deltas=source_adapter['settlement_deltas'],
        settlement_scores='per_event' if detailed else 'group_final',
        result_schema=revision if detailed else None,
        game_metadata_schema=revision if convlog else None,
        native_rules=native_rule_kwargs(profile),
        hand_rules=hand_rules_from_profile(profile),payment_rules=payment_rules_from_profile(profile),
        furiten_rules=furiten_rules_from_profile(profile),aka_counts=list(profile['rule_options']['aka_counts']),
        win_rules=win_rules_from_profile(profile),riichi_policy=riichi_policy_from_profile(profile),
        abort_rules=abort_rules_from_profile(profile),
        round_rules=dict(initial_scores=list(profile['initial_scores']),target_points=profile['target_points'],
            **profile['round_policy']))


def capability_report(profile):
    """Every strategy parameter is accounted for; mapped is not verified."""
    validate_profile(profile)
    rows=[]
    def visit(value,path):
        if type(value) is dict:
            for name,item in value.items():
                visit(item,f'{path}.{name}' if path else name)
        else:
            rows.append(dict(path=path,value=value,status='unverified',
                reason='full profile behavior has no independent capability acceptance yet'))
    for key,value in profile.items():
        if key not in ('ruleset_id','revision','platform','report_options'):
            visit(value,key)
    mapped={'rule_options.ankan_chankan','rule_options.multiple_ron','rule_options.kuicae',
            'rule_options.yakuman_multipliers','payment_policy.pao_scope','score_policy.dealer_first_discard'}
    for row in rows:
        if row['path'] in mapped:
            row.update(status='mapped_unverified',reason='explicit native GameRule mapping; behavior acceptance still required')
        if row['path'].startswith('round_policy.') or row['path']=='target_points':
            row.update(status='component_verified',reason='round-termination/1 finite boundary table; not historical admission',
                implementation='mahjong.envs.round_rules.RoundRules',
                fixtures='tests/unit/cleaning/test_round_termination.py')
    try:
        inventory_rules=inventory_rules_from_profile(profile)
    except ReplayError as error:
        inventory_rules=None
        for row in rows:
            if row['path']=='rule_options.aka_counts':
                row.update(status='unsupported',reason=error.reason)
    else:
        for row in rows:
            if row['path']=='rule_options.aka_counts':
                row.update(status='component_verified',reason='explicit ordinary/red observed stock capacities; full profile integration unverified',
                           implementation='mahjong.envs.tile_stock.TileStock',
                           fixtures='tests/unit/cleaning/test_tile_stock.py')
    try:
        hand_rules=hand_rules_from_profile(profile)
    except ReplayError as error:
        hand_rules=None
        for row in rows:
            if row['path']=='rule_options.riichi_kan_policy':
                row.update(status='unsupported',reason=error.reason)
    else:
        for row in rows:
            if row['path'] in {'rule_options.riichi_min_score','rule_options.riichi_min_live','rule_options.riichi_kan_policy'}:
                row.update(status='component_verified',reason='explicit hand checks; full profile integration still unverified',
                           implementation='mahjong.envs.hand_rules.HandRules',
                           fixtures='tests/unit/cleaning/test_hand_rules.py')
    try:
        mapping=native_rule_kwargs(profile)
    except ReplayError as error:
        mapping=None
        rows.append(dict(path='native_rule_mapping',value=None,status='unsupported',reason=error.reason))
    try:
        payment_rules=payment_rules_from_profile(profile)
    except ReplayError as error:
        payment_rules=None
        for row in rows:
            if row['path']=='rule_options.kazoe':
                row.update(status='unsupported',reason=error.reason)
    else:
        # These are component proofs with explicit result/source-policy facts,
        # not complete profile admission or a default for missing facts.
        paths={'rule_options.kazoe','rule_options.kiriage','payment_policy.honba_ron_each',
               'payment_policy.honba_tsumo_each','payment_policy.multi_ron_honba',
               'payment_policy.multi_ron_kyotaku','payment_policy.noten_total','payment_policy.final_kyotaku',
               'payment_policy.pao_scope','payment_policy.pao_ron','payment_policy.pao_tsumo',
               'payment_policy.nagashi','payment_policy.nagashi_noten'}
        for row in rows:
            if row['path'] in paths:
                row.update(status='component_verified',reason='explicit payment facts only; full source/profile integration unverified',
                           implementation='mahjong.envs.payment_rules.PaymentRules',
                           fixtures='tests/unit/cleaning/test_payment_rules.py')
                if row['path'] in {'payment_policy.pao_scope','payment_policy.pao_ron','payment_policy.pao_tsumo'}:
                    row.update(additional_fixtures=['tests/unit/cleaning/test_pao_results.py',
                        'tests/unit/cleaning/test_pao_honba.py'],
                        required_facts='verified liability; positive liability honba requires an explicit verified honba_payer')
                if row['path'] in {'payment_policy.nagashi','payment_policy.nagashi_noten'}:
                    row.update(additional_fixtures=['tests/unit/cleaning/test_draw_results.py',
                        'tests/unit/cleaning/test_nagashi_extras.py'],
                        required_facts='verified recipients/tenpai; nonzero extras require verified source-policy facts')
                if row['path']=='payment_policy.noten_total' and row['value']%12:
                    row.update(status='unsupported',reason='equal integer noten transfers require divisibility by 3 and 4')
    dora_rules=dora_rules_from_profile(profile)
    for row in rows:
        if row['path'].startswith('dora_timing.'):
            row.update(status='component_verified',reason='explicit FIFO reveal boundaries; full source/profile integration unverified',
                       implementation='mahjong.envs.offline_mjai.OfflineSession',
                       fixtures='tests/unit/cleaning/test_dora_timing.py')
    furiten_rules=furiten_rules_from_profile(profile)
    for row in rows:
        if row['path'].startswith('furiten_policy.'):
            row.update(status='component_verified',reason='explicit three-state response knowledge; source collection identity/full profile unverified',
                       implementation='mahjong.envs.furiten.FuritenKnowledge',fixtures='tests/unit/cleaning/test_furiten.py')
    return dict(native_kwargs=mapping,inventory_rules=inventory_rules,hand_rules=hand_rules,payment_rules=payment_rules,dora_rules=dora_rules,
                furiten_rules=furiten_rules,parameters=rows,production_ready=False)
