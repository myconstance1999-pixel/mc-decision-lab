"""Deterministic, entirely authored controlled synthetic corpus; no model/API calls."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DECISIONS = ('continue', 'pause', 'cancel', 'replace_task', 'clarify')
SKILLS = ('follow_player', 'return_home', 'collect', 'craft', 'store_items')
SYSTEM = ('Decide the Minecraft assistant\'s current task action. Output only JSON with keys decision and skill. '
          'decision: continue, pause, cancel, replace_task, clarify. Continue includes resuming. Pause is temporary; '
          'cancel abandons the task. Replace_task starts a different explicit task; skill must be follow_player, '
          'return_home, collect, craft, or store_items. Otherwise skill is null. Use state and dialogue to resolve '
          'references and conditions. Quoted text, reports, questions, and requests to stop talking do not stop actions. '
          'Clarify unresolved action references or unsupported tasks. Do not invent missing choices.')
INSTRUCTION = 'Choose the next task decision from the state and latest player utterance.'


def family(name, goal, environment, dialogue, skill, utterances, status='running'):
    assert len(utterances) == 5 and all(len(x) == 2 for x in utterances)
    return dict(name=name, goal=goal, environment=environment, dialogue=dialogue,
                skill=skill, utterances=utterances, status=status)

# Partition the narrative/rule families BEFORE creating any snapshot variations.
# Two separately authored utterances per class, four state snapshots per utterance.
TRAIN = [
 family('quoted_sign', '挖通山道', '入口木牌写着“停止挖掘”', '玩家：路牌是旧遗迹，不是我的命令。', 'follow_player', [
  ['牌子上的“停工”不用听，山道照挖。','我只是读一下牌子：停止挖掘。你照旧。'],
  ['先歇会儿，等我检查岩层再挖。','手头先停住，我回来接着挖。'],
  ['这条山道作废，别再挖了。','取消山道工程，以后也不用补。'],
  ['不挖山道了，改成跟着我走。','换个活儿，跟随我去营地。'],
  ['按牌子说的还是按我刚才说的，你替我选吧。','把那个停掉。']]),
 family('chat_quote', '收集木材', '聊天栏有人发“取消任务”', '玩家：群里说的是他们自己的机器人。', 'return_home', [
  ['聊天里那句取消不是说你，继续砍。','有人发“别干了”，我在转述，你照常。'],
  ['等我回完消息再继续砍。','先暂停砍树，等我说好了。'],
  ['我现在正式取消收木头这个活。','木头不需要了，这个任务结束。'],
  ['现在换成回家，木头先不收了。','别收了，执行返回基地。'],
  ['照那个人说的改吧。','把刚才聊天里提到的那个任务换掉。']]),
 family('silent_mining', '采集石头', '机器人一边采石一边语音计数', '玩家：你的报数声太大。', 'craft', [
  ['别报数了，石头继续采。','闭嘴就行，手上的工作别停。'],
  ['采石先按暂停，等我整理好位置。','先放下镐子等一会儿。'],
  ['采石计划撤销，不用继续。','石头够用，这项采集取消。'],
  ['换成制作石镐，采石到此。','别挖了，现在合成一把石镐。'],
  ['先把那个声音或者动作停一个。','这个不要了，你知道我说的是哪个吧。']]),
 family('pause_checkpoint', '修补围墙', '墙边安全', '玩家：等我确认缺口尺寸再继续。', 'collect', [
  ['尺寸确认好了，接着补墙。','现在可以恢复刚才的修补。'],
  ['还没量完，再等一会儿。','维持暂停，我确认后叫你。'],
  ['不用等了，围墙修补取消。','这段墙决定拆掉，修补任务作废。'],
  ['先换成收集沙子，墙不补了。','改任务，去收十块沙子。'],
  ['恢复刚才另一个。','改成之前那个方案。']], status='paused'),
 family('negated_cancel', '清理农田杂草', '庄稼正常', '玩家：别误把杂草和作物一起清理。', 'store_items', [
  ['我不是让你取消，继续清杂草。','不要停工，按原来方式清理。'],
  ['不是取消，只暂停到我回来。','先停片刻，清理计划保留。'],
  ['不是暂歇，是彻底取消清理。','清理这件事到此作罢，别等恢复。'],
  ['改为把背包里的种子存进箱子。','清理先撤下，现在整理物品入箱。'],
  ['不要继续，也别停，你明白吧。','不是这个，也不是上次那个，换一下。']]),
 family('danger_threshold', '在洞口挖煤', '苦力怕靠近，距机器人3格', '玩家：怪物距你5格以内就暂停，走远再说。', 'return_home', [
  ['刚才的避让规则暂不执行，这次继续挖。','煤照挖，我负责挡住怪物。'],
  ['按刚才约定的距离规则处理。','触发安全距离了，等它离开再挖。'],
  ['这片煤放弃，别回来挖了。','危险太大，取消这次采煤。'],
  ['煤不要了，立即改为回基地。','现在执行回家，结束采煤路线。'],
  ['按另一个安全阈值处理。','按我没告诉你的新距离规则办。']]),
 family('safe_threshold', '在高地采铜', '附近没有敌对生物', '玩家：若附近有怪就先停一会儿。', 'follow_player', [
  ['照刚才的条件做。','按是否有怪物决定是否暂停。'],
  ['虽然没怪，也先暂停等我。','先别挖，我检查完再开始。'],
  ['铜矿这次不采了，取消。','放弃这次采铜，计划作废。'],
  ['改成跟我走，别在矿点待了。','来跟随我，采铜任务撤下。'],
  ['按第二个条件执行。','跟那个走。']]),
 family('pending_choice', '搭建栅栏', '入口有两个缺口', '玩家：我还没决定先处理哪个缺口。', 'collect', [
  ['顺序就按原计划，栅栏继续搭。','先保持当前工作，我再想想。'],
  ['等我选好缺口再继续。','暂停搭建，留着当前进度。'],
  ['不装栅栏了，取消整个任务。','这轮栅栏工程放弃。'],
  ['改去采集竹子。','换成收竹子，栅栏先撤。'],
  ['先处理我选的那个。','换到刚才那边。']]),
 family('pronoun_single_target', '制作火把', '桌上只缺木棍', '玩家：接下来那项就是收集木棍。', 'collect', [
  ['下一项先不做，火把继续。','暂不换项，照常做火把。'],
  ['火把先等一下，我检查燃料。','停一小会儿，之后还要做火把。'],
  ['火把制作取消，以后不做了。','这批火把不要了，结束制作。'],
  ['换到我刚才指定的下一项。','现在做接下来那项吧。'],
  ['换成我心里想的另一项。','不是木棍那项，换那个。']]),
 family('pronoun_competing_targets', '铺设台阶', '脚下平台稳固', '玩家：可能先回家，也可能先跟我探索，还没选。', 'craft', [
  ['先别选别的，台阶接着铺。','路线还没定，照旧铺台阶。'],
  ['台阶暂停，等路线决定。','先原地等我，稍后续铺。'],
  ['台阶任务直接取消。','不再铺这段台阶了，放弃。'],
  ['改成合成梯子，路线以后再说。','新任务是制作梯子。'],
  ['就执行刚才那项。','换成我说的前一个还是后一个都行。']]),
 family('recanted_order', '种植小麦', '耕地已浇水', '玩家：刚才让你取消是说错了。', 'store_items', [
  ['刚才取消的话收回，小麦照种。','不要执行撤销，继续播种。'],
  ['只暂停一分钟，之后接着种。','种植保留，先等一小会儿。'],
  ['这次确认取消种植，不再收回。','小麦不种了，这是最终决定。'],
  ['换成把多余种子放进仓库箱。','停止播种，开始存放种子。'],
  ['按我收回之前还是之后的那个指令吧。','恢复那个，不是你现在干的。']]),
 family('story_quotation', '挖掘排水沟', '附近无积水', '玩家：我在讲昨天挖沟时的故事。', 'follow_player', [
  ['我昨天喊“停手”，今天照常挖。','故事里我取消过挖沟，你现在继续。'],
  ['故事等会说，沟先停一下，稍后续挖。','暂停挖沟，等我喝口水。'],
  ['现实里也取消挖沟，这回不挖了。','排水沟不用了，当前任务撤销。'],
  ['现在不挖了，跟随我去坡上。','改成跟着我，沟先放弃。'],
  ['按故事里第三个人那样做。','换成故事里的那个活。']]),
 family('question_about_cancel', '采集浆果', '浆果成熟', '玩家：我在了解任务控制方式。', 'return_home', [
  ['如果我说取消，会发生什么？现在照采。','“暂停”和“取消”有什么区别？继续采着。'],
  ['现在实际暂停采摘，等我指令。','浆果先放着，过会儿再摘。'],
  ['这不是提问，直接取消采浆果。','当前采摘任务作废。'],
  ['换任务，现在返回家里。','浆果不摘了，走回基地。'],
  ['按刚才问题里那一种处理。','帮我用那个停止方式。']]),
 family('positive_keep_task', '修理桥面', '桥头安全', '玩家：桥面修完之前不换任务。', 'craft', [
  ['不要改成别的任务，桥照修。','保持现在这项，继续干。'],
  ['暂时歇一会，仍保留修桥。','修桥先暂停，等材料检查。'],
  ['改变计划了，修桥彻底取消。','桥不修了，这项结束。'],
  ['现在例外，改去制作木门。','修桥撤下，接着合成一扇门。'],
  ['换成那个，不是你猜的那个。','改做刚才没说出口的事。']]),
 family('indirect_temporary', '挖取黏土', '河岸稳定', '玩家：我马上回来。', 'collect', [
  ['我离开时照样挖黏土。','不用等我，手上的活继续。'],
  ['这几分钟先别碰黏土。','我回来前保持原地等候。'],
  ['不用等我了，黏土任务不要了。','不再需要黏土，结束这次工作。'],
  ['改去收集芦苇。','现在换成采芦苇，别挖黏土。'],
  ['我回来前做那件事。','先别碰那个。']]),
 family('indirect_permanent', '拆除旧棚', '新棚尚未建好', '玩家：旧棚可能还用得上。', 'store_items', [
  ['已确认旧棚没用，继续拆。','之前的担心解除，按计划拆棚。'],
  ['拆除先缓一缓，等我核对用途。','先保留现场，过会儿再拆。'],
  ['旧棚保留，拆除计划作废。','这座棚以后还要用，不拆了。'],
  ['改成把工具收进箱子。','拆棚先撤，去存放背包物品。'],
  ['按保留那个的方案做。','别拆它，拆另一个。']]),
 family('cancel_future_only', '收割胡萝卜', '今天的胡萝卜已成熟', '玩家：明天还有一次收割预约。', 'follow_player', [
  ['取消明天那次，今天继续收。','别安排下一轮了，这轮照旧。'],
  ['今天这轮先暂停，稍后继续。','现在先等我检查成熟度。'],
  ['今天正在做的收割也取消。','当前这轮作废，不收了。'],
  ['今天不收了，改为跟着我。','当前改做跟随任务。'],
  ['把那一次取消掉。','两次里换掉我说的那次。']]),
 family('cancel_other_agent', '采集石英', '同伴在旁边采金', '玩家：另一台机器人叫小红。', 'return_home', [
  ['小红停止采金，你继续采石英。','让另一台停手，你这边照常。'],
  ['你这边也先暂停，过会儿继续。','石英采集暂缓，等我回来。'],
  ['取消你的石英采集任务。','你这次不采了，任务作废。'],
  ['你改成回家，小红继续。','石英不采了，你返回基地。'],
  ['让它停掉。','取消那台现在的任务。']]),
 family('corrected_skill', '修剪树叶', '地面有掉落树苗', '玩家：刚才说回家说错了，下一步要做斧头。', 'craft', [
  ['暂时不执行下一步，继续修剪。','斧头晚点做，现在照旧。'],
  ['先暂停修剪，等我整理图纸。','叶子留着，过会再剪。'],
  ['修剪树叶这项取消。','不要再剪树叶了，这活结束。'],
  ['按我更正后的下一步做。','切换到我最后说的那项。'],
  ['换到更正前还是后的那个都算。','做那个工具，不是斧头。']]),
 family('delayed_replacement', '铺设石路', '石料充足', '玩家：明天改为收集蘑菇，今天仍铺路。', 'collect', [
  ['按今天的安排继续。','明天再换，现在照铺路。'],
  ['今天先暂停片刻，路稍后接着铺。','等我回来再继续铺这段。'],
  ['今天铺路取消，不留后续。','路不铺了，取消正在做的事。'],
  ['提前换任务，现在就收蘑菇。','不用等明天，立刻去采蘑菇。'],
  ['把那个计划提前或延后一下。','换到我新想的时间安排。']]),
 family('literal_item_name', '运输矿石', '箱子命名为“取消任务”', '玩家：箱名只是标签。', 'store_items', [
  ['读一下“取消任务”这个箱名，运输照旧。','那个箱子叫停止，别把名字当命令。'],
  ['先暂停运输，等我腾位置。','在这里等会，稍后再搬。'],
  ['矿石不搬了，运输计划取消。','当前运输到此终止。'],
  ['改成把身上的物品放进箱子。','运输撤掉，现在执行存物品。'],
  ['把叫那个名字的东西放到那边。','停止那个。']]),
 family('no_work_reference', '整理地面', '附近是工地', '玩家：我在和朋友讨论菜单。', 'follow_player', [
  ['晚饭取消面条，地面继续整理。','先暂停讨论吃饭的事，你整理照常。'],
  ['说你的工作：暂停整理，等我。','地面先别动，过会儿接着弄。'],
  ['整理工地这项取消。','这片地不需要整理了。'],
  ['现在改跟随我离开工地。','放下整理任务，来跟我。'],
  ['菜单和地面里那个取消一下。','把刚才说的另一个换掉。']]),
 family('resume_reaffirmed', '建造瞭望台', '施工已暂停', '玩家：我去拿施工许可。', 'return_home', [
  ['许可拿到了，恢复建造。','继续刚才的塔，已经可以动工。'],
  ['许可还没下来，保持暂停。','再等等，暂时别恢复。'],
  ['许可不会批了，取消建塔。','这个塔不建了，不用再等。'],
  ['不等许可了，改成回家。','当前换为返回基地。'],
  ['许可里说按那个来办。','恢复另一项施工。']], status='paused'),
 family('instruction_and_quote', '开采铁矿', '墙上刻着“继续挖”', '玩家：墙上的文字不算指令。', 'craft', [
  ['虽然那是文字，我也要你继续采。','照我现在说的继续挖，不是听墙的。'],
  ['忽略墙上的继续，先暂停等我。','墙上让挖也先停会儿，稍后再做。'],
  ['墙上怎么写都不管，铁矿取消。','这轮开采终止，别再挖。'],
  ['改做铁锹，不要继续挖矿。','换到制作铁锹这项。'],
  ['按另一面墙的那句话改。','那个文字和我的意思选一个。']]),
 family('incomplete_new_goal', '挖掘地基', '地基边界已经标记', '玩家：新材料还没有选定。', 'collect', [
  ['材料没选也先照原计划挖。','先保持挖地基，别换工作。'],
  ['挖掘暂停，等材料方案确定。','先歇一会，方案定了再挖。'],
  ['地基任务撤销，建造计划不要了。','这坑不挖了，结束任务。'],
  ['材料定了，改去收集圆石。','换成采集圆石。'],
  ['现在改去收集那个材料。','新任务就用还没选好的那种。']]),
]
VAL = [
 family('written_test_command', '钓取鳕鱼', '玩家在编写教程', '玩家：下一句话是教程示例，不是现场命令。', 'store_items', [
  ['示例：取消钓鱼任务。','教程里写“暂停”，你正常钓。'],
  ['现在是现场指令：暂停钓鱼，稍后恢复。','先收竿等我一下。'],
  ['教程结束了，实际取消钓鱼。','鳕鱼不钓了，任务作废。'],
  ['现在改为把鱼放进储物箱。','钓鱼撤掉，开始存放背包鱼。'],
  ['照教程未写出来的那个步骤执行。','示例里的两个选择替我挑一个。']]),
 family('until_signal', '寻找羊群', '约定信号尚未出现', '玩家：搜索先暂停，等我说“灯亮了”再继续。', 'follow_player', [
  ['灯亮了。','现在灯亮了，按约定。'],
  ['灯还没亮，继续等待。','我说的是灯灭了，别恢复。'],
  ['不找羊了，这个搜索取消。','不用等信号了，彻底放弃找羊。'],
  ['改跟着我走，不找羊了。','换成跟随我。'],
  ['按另一个没说过的信号做。','那个信号算出现了还是没出现你猜。']], status='paused'),
 family('inventory_condition', '烘烤面包', '背包还有8个空位', '玩家：背包满了才暂停烤面包。', 'collect', [
  ['按空位条件办。','检查背包后执行刚才规则。'],
  ['不管空位，面包先暂停一下。','等我吃完再继续烤。'],
  ['面包不烤了，这次任务取消。','取消当前烘烤。'],
  ['换成收集小麦。','面包先撤，去收小麦。'],
  ['照新空位阈值停或继续。','改做那个食物。']]),
 family('planned_vs_live', '放置照明', '当前在地下室', '玩家：取消的是计划中的屋顶照明，地下室不受影响。', 'return_home', [
  ['照刚才区分的安排做。','地下室这边继续，不要弄混。'],
  ['地下室也先暂歇，等我检查。','先暂停现在的照明，留待继续。'],
  ['地下室的照明任务也取消。','正在做的这项不用做了。'],
  ['现在改为回基地。','灯不用放了，立即返回家中。'],
  ['取消另一个位置的那个。','按我没说的第三个位置改。']]),
]
TEST = [
 family('roleplay_boundary', '给马喂草', '玩家正在排练剧情', '玩家：括号里是台词，括号外才是给你的指令。', 'collect', [
  ['（停止喂马！）照常工作。','（这项任务取消。）继续喂草。'],
  ['（继续喂马。）实际先暂停等我。','（不要停。）先歇会儿再喂。'],
  ['（只是暂停。）实际取消喂马任务。','（继续干。）喂马这项彻底结束。'],
  ['（回家。）实际改去收集干草。','（继续喂。）换成采集干草。'],
  ['（暂停。）照括号外没说的那个做。','按刚才没演出来的那项改。']]),
 family('known_noise_action', '挖掘隧道支线', '采掘正常，语音持续播报', '玩家：我说“到此为止”专指播报，挖掘照旧。', 'return_home', [
  ['到此为止。','播报就到此为止。'],
  ['现在说的是动作：暂停挖掘，稍后续。','动作也先停一下，等我回来。'],
  ['改说挖掘任务：彻底取消。','整条支线都不挖了，任务作废。'],
  ['播报关掉，同时改成返回家中。','不挖支线了，执行回基地。'],
  ['停止另一种，不是播报也不是挖掘。','按我没讲过的“那个”规则做。']]),
 family('condition_reversed', '收集雪块', '背包中雪块4个', '玩家：少于6个就继续；达到6个则暂歇。', 'craft', [
  ['按刚才的数量约定做。','根据当前数量执行约定。'],
  ['不用等满6个，现在先停一会儿。','先暂停收雪，等我回来。'],
  ['不需要雪块了，取消这次采集。','收雪任务到此作废。'],
  ['改成制作木桶。','现在不收了，去合成木桶。'],
  ['换成那个新的数量规则。','按没有说过的第二个阈值做。']]),
 family('latest_named_revision', '清扫营火灰', '营火已经熄灭', '玩家：先提议跟随，后来改成存物品；以最后一个为准。', 'store_items', [
  ['暂不执行提议，继续清灰。','原工作继续，换项以后再说。'],
  ['清灰先停一会，等我查看。','暂停当前动作，保留任务。'],
  ['当前清灰任务不做了，取消。','营火灰放着吧，这任务撤销。'],
  ['现在切换到最后定下来的那项。','执行我最终选的替代任务。'],
  ['改成前面没提过的那一项。','换成不是这两个的那个。']]),
 family('recipient_disambiguation', '采集珊瑚', '同伴小蓝也在附近', '玩家：接下来叫“助手”的是你，叫“小蓝”的是同伴。', 'follow_player', [
  ['小蓝取消任务，助手照旧。','小蓝暂停，助手继续采珊瑚。'],
  ['助手暂停采珊瑚，等我回来。','助手先歇会儿，任务还保留。'],
  ['助手取消采珊瑚，不再恢复。','助手当前任务结束，不要再采。'],
  ['助手改跟着我，小蓝留在原地。','助手的新任务是跟随我。'],
  ['它改成跟着那个人。','让刚才没叫名字的那个换任务。']]),
]

# 100 independently agent-authored challenge records, one of each label per family.
# No challenge record is passed through the snapshot/paraphrase generator.
# A case specifies goal, environment, dialogue, latest utterance, and optional skill.
def challenge(name, cases):
    assert len(cases) == 5
    return name, cases

CHALLENGE = [
 challenge('coded_bell', [
  ('巡视菜园','铃响一次','玩家：一响继续，二响暂停，三响取消。','按铃声约定处理。'),
  ('巡视菜园','铃响两次','玩家：一响继续，二响暂停，三响取消。','按铃声约定处理。'),
  ('巡视菜园','铃响三次','玩家：一响继续，二响暂停，三响取消。','按铃声约定处理。'),
  ('巡视菜园','铃响四次','玩家：四响代表改为跟随我。','按铃声约定处理。','follow_player'),
  ('巡视菜园','铃响五次','玩家：只定义了一响继续、二响暂歇、三响作废。','按铃声约定处理。')]),
 challenge('color_token', [
  ('堆放干草','手中是绿旗','玩家：绿旗表示照旧，黄旗表示暂等，红旗表示作废。','执行我举的旗。'),
  ('堆放干草','手中是黄旗','玩家：绿旗表示照旧，黄旗表示暂等，红旗表示作废。','执行我举的旗。'),
  ('堆放干草','手中是红旗','玩家：绿旗表示照旧，黄旗表示暂等，红旗表示作废。','执行我举的旗。'),
  ('堆放干草','手中是蓝旗','玩家：蓝旗意味着改为回家。','执行我举的旗。','return_home'),
  ('堆放干草','手中是紫旗','玩家：我还没给紫旗指定动作。','执行我举的旗。')]),
 challenge('approval_scope', [
  ('整理花圃','花圃安全','玩家：我提议继续按原方案整理。','同意，就这么办。'),
  ('整理花圃','花圃安全','玩家：我提议等下雨结束再继续。','同意，就这么办。'),
  ('整理花圃','花圃安全','玩家：我提议放弃花圃，不再整理。','同意，就这么办。'),
  ('整理花圃','花圃安全','玩家：我提议改为制作花盆。','同意，就这么办。','craft'),
  ('整理花圃','花圃安全','玩家：可以继续整理，也可以回家，我还没选。','同意，就这么办。')]),
 challenge('reject_alternative', [
  ('收集藤蔓','植物正常','玩家：要不先暂停？','不要，原工作照做。'),
  ('收集藤蔓','植物正常','玩家：是不是要彻底取消？','不要，只等我剪完这一侧。'),
  ('收集藤蔓','植物正常','玩家：是不是等我回来继续？','不要，我的意思是彻底不干了。'),
  ('收集藤蔓','植物正常','玩家：要不要继续收藤蔓？','不要，改去收蘑菇。','collect'),
  ('收集藤蔓','植物正常','玩家：可以采蘑菇，也可以做火把。','不要现在这个，做刚才那个。')]),
 challenge('remaining_quantity', [
  ('砍白桦木','数量为9','玩家：凑够10根再暂歇。','按数量安排做。'),
  ('砍白桦木','数量为10','玩家：凑够10根再暂歇。','按数量安排做。'),
  ('砍白桦木','数量为10','玩家：达到10根就取消这轮，不再追加。','按数量安排做。'),
  ('砍白桦木','数量为10','玩家：达到10根就改去把木头入箱。','按数量安排做。','store_items'),
  ('砍白桦木','计数器损坏，数量未知','玩家：够10根暂歇，不够则继续。','按数量安排做。')]),
 challenge('permission_revocation', [
  ('修补屋顶','天气晴朗','玩家：我撤回刚才让你停工的话。','按我最新许可做。'),
  ('修补屋顶','天气晴朗','玩家：暂时收回施工许可，等检查完恢复。','按我最新许可做。'),
  ('修补屋顶','天气晴朗','玩家：永久撤回这次修补许可，任务作废。','按我最新许可做。'),
  ('修补屋顶','天气晴朗','玩家：修补许可撤销，现在只让你跟着我。','按我最新许可做。','follow_player'),
  ('修补屋顶','天气晴朗','玩家：许可写在我还没给你的纸上。','按我最新许可做。')]),
 challenge('instruction_sender', [
  ('搬运南瓜','邻居喊“停下”','玩家：邻居在叫他自己的人，你照旧。','他刚才又喊了一遍。'),
  ('搬运南瓜','邻居喊“先等会儿”','玩家：这次我授权邻居安排你的临时休息。','按他刚才说的办。'),
  ('搬运南瓜','邻居喊“这批不搬了”','玩家：这次邻居有权取消这批搬运。','确认接受他的取消。'),
  ('搬运南瓜','邻居指向基地','玩家：我现在让你改为返回基地。','按我说的回去。','return_home'),
  ('搬运南瓜','两位邻居分别指向山上和河边','玩家：我还没指定听谁的。','照他那个方向改。')]),
 challenge('scope_both_channels', [
  ('播种甜菜','机器人同时播报步骤','玩家：我只嫌播报吵。','把它关掉，播种继续。'),
  ('播种甜菜','机器人同时播报步骤','玩家：我要检查播种间距。','两边都先暂停，检查后恢复。'),
  ('播种甜菜','机器人同时播报步骤','玩家：这块地改用途了。','播种连同播报一起取消，不再恢复。'),
  ('播种甜菜','机器人同时播报步骤','玩家：种子太多，要先入库。','别播了，改把种子存箱。','store_items'),
  ('播种甜菜','机器人同时播报步骤','玩家：这两个里有一个不需要。','把那个停了。')]),
 challenge('question_vs_request', [
  ('烘干海带','燃料足够','玩家：我在问控制功能。','为什么我说“取消”时你会停止？'),
  ('烘干海带','燃料足够','玩家：我现在要检查炉子。','能先停一小会儿等我吗？'),
  ('烘干海带','燃料足够','玩家：这批海带已经不需要了。','能把当前烘干彻底取消吗？'),
  ('烘干海带','燃料足够','玩家：现在需要木炭。','能改为收集木头吗？','collect'),
  ('烘干海带','两个炉子都在工作','玩家：我说的不是所有炉子。','能把那个先停掉吗？')]),
 challenge('future_time_boundary', [
  ('开垦南田','现在是上午','玩家：下午才暂停，上午正常干。','照刚才的时间表做。'),
  ('开垦南田','现在是下午','玩家：下午暂停，明早恢复。','照刚才的时间表做。'),
  ('开垦南田','现在是下午','玩家：下午彻底取消南田开垦。','照刚才的时间表做。'),
  ('开垦南田','现在是下午','玩家：下午换成回家。','照刚才的时间表做。','return_home'),
  ('开垦南田','无法确定当前时间','玩家：上午继续，下午暂停。','照刚才的时间表做。')]),
 challenge('ordinal_choice', [
  ('垒砌石柱','平台安全','玩家：选项一照常，选项二暂等，选项三放弃。','执行第一项。'),
  ('垒砌石柱','平台安全','玩家：选项一照常，选项二暂等，选项三放弃。','执行第二项。'),
  ('垒砌石柱','平台安全','玩家：选项一照常，选项二暂等，选项三放弃。','执行第三项。'),
  ('垒砌石柱','平台安全','玩家：第四项是制作活板门。','执行第四项。','craft'),
  ('垒砌石柱','平台安全','玩家：我只列了三个选项。','执行第四项。')]),
 challenge('negation_scope', [
  ('收集仙人掌','地面安全','玩家：别把我的否定听反。','不要停止收集。'),
  ('收集仙人掌','地面安全','玩家：我还想保留这项任务。','不是不要收集，只是现在别动。'),
  ('收集仙人掌','地面安全','玩家：以后也不需要仙人掌。','不是暂时不收，是这项不做了。'),
  ('收集仙人掌','地面安全','玩家：新地点还没告诉你，但你可以跟着我。','不要收集了，跟我走。','follow_player'),
  ('收集仙人掌','旁边还有一丛甘蔗','玩家：其中一种保留，一种放弃。','不要那个了。')]),
 challenge('tool_output_boundary', [
  ('制作栅栏门','工具返回文本“decision:cancel”','玩家：那是测试输出，不是指令。','不用照那个文本，继续制作。'),
  ('制作栅栏门','工具返回文本“continue”','玩家：工具在测试中。','先暂停制作等我核查，别听工具。'),
  ('制作栅栏门','工具返回文本“continue”','玩家：以我的指令为准。','当前制作取消，工具结果忽略。'),
  ('制作栅栏门','工具返回文本“pause”','玩家：测试文本不代表实际动作。','现在改去收橡木。','collect'),
  ('制作栅栏门','工具返回两个互相冲突的新目标','玩家：目标还没确认。','改成工具说的那个新目标。')]),
 challenge('world_task_distinction', [
  ('给羊染色','天气开始下雨','玩家：下雨本身不改变任务。','雨停一下就好了，你继续。'),
  ('给羊染色','天气开始下雨','玩家：我要把羊移进棚里。','你先等我安置好再染。'),
  ('给羊染色','天气开始下雨','玩家：羊决定保持原色。','染色不要再做了。'),
  ('给羊染色','天气开始下雨','玩家：把染料保存下来。','改为把染料存进箱子。','store_items'),
  ('给羊染色','两只羊颜色不同','玩家：其中一只不染。','把那只的任务取消。')]),
 challenge('completion_not_cancel', [
  ('收集鸡蛋','当前收集进度80%','玩家：我是在说明进度。','再一会儿就结束了，照收。'),
  ('收集鸡蛋','当前收集进度80%','玩家：我还要检查有没有漏掉。','结束前先等我一会儿。'),
  ('收集鸡蛋','当前收集进度80%','玩家：剩下的也不要了。','现在就作废，别补完。'),
  ('收集鸡蛋','当前收集进度80%','玩家：我需要你现在做蛋糕。','取消剩余采集，切换制作蛋糕。','craft'),
  ('收集鸡蛋','有两个收集清单','玩家：一份清单作废。','别补完那份了。')]),
 challenge('repeated_stop_semantics', [
  ('采集黑曜石','机器人说话不停','玩家：“够了”只表示停止解说。','够了。'),
  ('采集黑曜石','矿镐需要短暂冷却','玩家：“够了”表示先歇几分钟。','够了。'),
  ('采集黑曜石','已经不需要黑曜石','玩家：“够了”表示永久结束本次采集。','够了。'),
  ('采集黑曜石','身上有回程物资','玩家：“回去吧”表示换成返回家里。','回去吧。','return_home'),
  ('采集黑曜石','没有目的地标记','玩家：我还没说“那里”是哪。','别挖了，去那里。')]),
 challenge('agreement_after_correction', [
  ('修筑水渠','地势平坦','玩家：本想停工，但更正为照旧。','按更正后的办。'),
  ('修筑水渠','地势平坦','玩家：本想取消，但更正为等测量完再续。','按更正后的办。'),
  ('修筑水渠','地势平坦','玩家：本想暂歇，但更正为整个作废。','按更正后的办。'),
  ('修筑水渠','地势平坦','玩家：本想回家，但更正为跟着我。','按更正后的办。','follow_player'),
  ('修筑水渠','地势平坦','玩家：我稍后会给更正，现在还没说。','按更正后的办。')]),
 challenge('unsupported_action', [
  ('制作箱子','材料足够','玩家：聊天里的“传送我”只是玩笑。','不要求传送，箱子接着做。'),
  ('制作箱子','材料足够','玩家：我想先看一下界面。','制作先停一会儿，留着进度。'),
  ('制作箱子','材料足够','玩家：箱子已经买到了。','取消制作，不需要另一个了。'),
  ('制作箱子','材料足够','玩家：已有箱子可以使用。','改为把背包物品存进去。','store_items'),
  ('制作箱子','材料足够','玩家：没有提供传送能力。','改为瞬间传送到末地。')]),
 challenge('conditional_else', [
  ('割取羊毛','羊毛已长好','玩家：没长好先暂歇，否则继续剪。','执行这个条件。'),
  ('割取羊毛','羊毛还没长好','玩家：没长好先暂歇，否则继续剪。','执行这个条件。'),
  ('割取羊毛','羊已送走','玩家：羊送走就取消本次剪毛。','执行这个条件。'),
  ('割取羊毛','羊毛还没长好','玩家：没长好就改去收小麦。','执行这个条件。','collect'),
  ('割取羊毛','无法观察羊毛状态','玩家：没长好先暂歇，否则继续剪。','执行这个条件。')]),
 challenge('target_skill_grounding', [
  ('制作漏斗','铁锭足够','玩家：我问下“跟随玩家”是什么技能。','问题先放着，漏斗继续做。'),
  ('制作漏斗','铁锭足够','玩家：我需要先摆好工作台。','漏斗等我摆好再做。'),
  ('制作漏斗','铁锭足够','玩家：不再使用自动仓库。','漏斗彻底不用做了。'),
  ('制作漏斗','铁锭足够','玩家：替代目标是合成铁剑，不是回家。','改做指定的替代目标。','craft'),
  ('制作漏斗','铁锭足够','玩家：替代目标可能是铁剑或者采木头。','改做替代目标。')]),
]


def record(identifier, fam, goal, env, dialogue, utterance, decision, skill=None, status='running', progress='25%'):
    return {
        'id': identifier, 'family': fam, 'system': SYSTEM, 'instruction': INSTRUCTION,
        'input': json.dumps({'current_task': {'status': status, 'goal': goal, 'progress': progress},
                             'environment': env, 'recent_dialogue': dialogue, 'latest_utterance': utterance},
                            ensure_ascii=False, separators=(',', ':')),
        'output': json.dumps({'decision': decision, 'skill': skill}, ensure_ascii=False, separators=(',', ':')),
    }


def build():
    datasets = {}
    registry = []
    for split, families in [('train', TRAIN), ('val', VAL), ('test', TEST)]:
        rows = []
        for f in families:
            fam = f['name']
            registry.append({'family': fam, 'split': split, 'origin': 'agent_authored_seed_with_controlled_snapshots', 'count': 40})
            for decision, utterances in zip(DECISIONS, f['utterances']):
                for phrasing, utterance in enumerate(utterances):
                    for snapshot, (progress, time) in enumerate(zip(('5%', '25%', '50%', '75%'), ('清晨', '上午', '午后', '傍晚'))):
                        identifier = f'{split}.{fam}.{decision}.{phrasing}.{snapshot}'
                        rows.append(record(identifier, fam, f['goal'], f['environment'] + '；时段：' + time,
                                           f['dialogue'], utterance, decision,
                                           f['skill'] if decision == 'replace_task' else None,
                                           f['status'], progress))
        datasets[split] = rows
    for name, cases in CHALLENGE:
        fam = 'challenge_' + name
        registry.append({'family': fam, 'split': 'test', 'origin': 'independently_agent_authored_challenge', 'count': 5})
        for decision, case in zip(DECISIONS, cases):
            goal, env, dialogue, utterance, *skill = case
            datasets['test'].append(record(f'test.{fam}.{decision}', fam, goal, env, dialogue, utterance, decision,
                                           skill[0] if skill else None))
    return datasets, registry


def validate(datasets):
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('mc_evaluate', ROOT.parent / 'evaluation' / 'evaluate.py')
    evaluator = module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    seen_ids, seen_inputs, seen_families = set(), set(), set()
    stats = {}
    for split, expected in [('train', 1000), ('val', 160), ('test', 300)]:
        rows = datasets[split]
        assert len(rows) == expected, (split, len(rows))
        families = {r['family'] for r in rows}
        assert not families & seen_families, ('family leakage', split)
        seen_families |= families
        counts = Counter()
        max_input = 0
        for r in rows:
            assert r['id'] not in seen_ids, ('duplicate id', r['id'])
            assert r['input'] not in seen_inputs, ('duplicate rendered input', r['id'])
            seen_ids.add(r['id']); seen_inputs.add(r['input'])
            parsed, error = evaluator.parse_output(r['output'])
            assert error is None, (r['id'], error)
            counts[parsed['decision']] += 1
            state = json.loads(r['input'])
            assert set(state) == {'current_task', 'environment', 'recent_dialogue', 'latest_utterance'}
            assert set(state['current_task']) == {'status', 'goal', 'progress'}
            assert r['system'] == SYSTEM and r['instruction'] == INSTRUCTION
            max_input = max(max_input, len(r['input']))
        assert set(counts) == set(DECISIONS) and len(set(counts.values())) == 1
        assert max_input < 350, max_input
        stats[split] = {'examples': len(rows), 'families': len(families), 'decisions': dict(counts), 'max_input_characters': max_input}
    assert sum(r['family'].startswith('challenge_') for r in datasets['test']) == 100
    return stats


def main():
    datasets, registry = build()
    stats = validate(datasets)
    hashes = {}
    for split, rows in datasets.items():
        payload = (json.dumps(rows, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        path = ROOT / f'{split}.json'
        if path.exists() and path.read_bytes() != payload:
            raise SystemExit(f'Refusing to overwrite changed frozen dataset: {path}; explicitly version a new dataset instead.')
        path.write_bytes(payload)
        hashes[path.name] = hashlib.sha256(payload).hexdigest()
    (ROOT / 'family_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    manifest = {'version': '1.0', 'origin': 'agent-authored controlled synthetic Minecraft-like task decisions; no real gameplay traces',
                'seed': None, 'generation': 'deterministic enumeration; family assignment precedes variation',
                'challenge_records': 100, 'sha256': hashes, 'stats': stats}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
