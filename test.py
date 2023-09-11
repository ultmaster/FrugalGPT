import copy
import sys

import dotenv
dotenv.load_dotenv()

sys.path.insert(0, "./src")

import FrugalGPT


dev = [['Q: april gold down 20 cents to settle at $1,116.10/oz\nA:', 'down','0'],
       ['Q: gold suffers third straight daily decline\nA:', 'down','1'],
       ['Q: Gold futures edge up after two-session decline\nA:', 'up','2'],
       ['Q: Dec. gold climbs $9.40, or 0.7%, to settle at $1,356.90/oz\nA:','up','3'],
       ['Q: Gold struggles; silver slides, base metals falter\nA:','up','4'],
       ['Q: feb. gold ends up $9.60, or 1.1%, at $901.60 an ounce\nA:','up','5'],
        ['Q: dent research : is gold\'s day in the sun coming soon?\nA:','none','6']
      ]
prefix = open('config/prompt/HEADLINES/prefix_e8.txt').read()
raw_data = copy.deepcopy(dev)
data = FrugalGPT.formatdata(dev,prefix)

MyCascade = FrugalGPT.LLMCascade()
MyCascade.load(loadpath="strategy/HEADLINES/",budget=0.000665)

index = 2
query = data[index][0]
query_raw = raw_data[index][0]
genparams=FrugalGPT.GenerationParameter(max_tokens=50, temperature=0.1, stop=['\n'])
answer = MyCascade.get_completion(query=query,genparams=genparams)
cost = MyCascade.get_cost()
print("query:",query_raw)
print("FrugalGPT LLMCascade answer:",answer)
