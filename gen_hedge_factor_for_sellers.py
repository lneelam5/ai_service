from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
import json
import os
import re
import sys
from typing_extensions import TypedDict
from typing import Optional, List, Dict
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

load_dotenv(override=True)

# Step 1. Define LLM (lazy initialization to avoid blocking on import)
print("Initializing ChatBedrock LLM...")
try:
    llm = ChatBedrock(
        model_id="amazon.nova-lite-v1:0",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        max_tokens=8192,
        model_kwargs={"temperature": 0.7, "maxTokens": 8192},
    )
    print("ChatBedrock LLM initialized successfully")
except Exception as e:
    print(f"Error initializing ChatBedrock: {e}", file=sys.stderr)
    raise


# Step 2. Define mocked seller data
mocked_sellers: List[Dict[str, float]] = [
#   {"Id": "5829431", "rt": 0.72}, {"Id": "7495812", "rt": 1.0}, {"Id": "2918457", "rt": 0.0}, {"Id": "6321548", "rt": 0.58}, {"Id": "8412570", "rt": 0.11},
#   {"Id": "7584213", "rt": 1.00}, {"Id": "4598123", "rt": 0.43}, {"Id": "1928564", "rt": 0.37}, {"Id": "5784312", "rt": 0.97}, {"Id": "8942176", "rt": 0.18},
#   {"Id": "4518972", "rt": 0.34}, {"Id": "3672481", "rt": 0.22}, {"Id": "5782934", "rt": 0.85}, {"Id": "2984715", "rt": 0.79}, {"Id": "6598712", "rt": 0.16},
#   {"Id": "8419572", "rt": 0.88}, {"Id": "5221893", "rt": 0.57}, {"Id": "1365942", "rt": 0.41}, {"Id": "8425179", "rt": 0.52}, {"Id": "9824156", "rt": 0.14},
#   {"Id": "7148392", "rt": 0.92}, {"Id": "2548793", "rt": 0.67}, {"Id": "6283719", "rt": 0.05}, {"Id": "4852719", "rt": 0.32}, {"Id": "1947836", "rt": 0.58},
#   {"Id": "8759432", "rt": 0.83}, {"Id": "9237154", "rt": 0.20}, {"Id": "5914378", "rt": 0.61}, {"Id": "2148795", "rt": 0.45}, {"Id": "4512673", "rt": 0.89},
#   {"Id": "7146529", "rt": 0.67}, {"Id": "5432918", "rt": 0.56}, {"Id": "9271543", "rt": 0.10}, {"Id": "7834159", "rt": 0.91}, {"Id": "6327198", "rt": 0.85},
#    {"Id": "7298754", "rt": 0.21}, {"Id": "8214756", "rt": 0.54}, {"Id": "6927518", "rt": 0.17}, {"Id": "8173245", "rt": 0.88}, {"Id": "3259841", "rt": 0.78},
#    {"Id": "9183275", "rt": 0.36}, {"Id": "4729851", "rt": 0.62}, {"Id": "5839247", "rt": 0.47}, {"Id": "1267945", "rt": 0.09}, {"Id": "4198632", "rt": 0.81},
#    {"Id": "5824891", "rt": 0.64}, {"Id": "7192845", "rt": 0.42}, {"Id": "7814356", "rt": 0.12}, {"Id": "6238947", "rt": 0.93}, {"Id": "7824596", "rt": 0.58},
#   {"Id": "2498751", "rt": 0.19}, {"Id": "6731829", "rt": 0.25}, {"Id": "3491827", "rt": 0.55}, {"Id": "2849167", "rt": 0.90}, {"Id": "9851742", "rt": 0.35},
#   {"Id": "4573189", "rt": 0.74}, {"Id": "6214793", "rt": 0.65}, {"Id": "9152748", "rt": 0.08}, {"Id": "7482195", "rt": 0.91}, {"Id": "2915748", "rt": 0.47},
#   {"Id": "4723816", "rt": 0.43}, {"Id": "5837294", "rt": 0.21}, {"Id": "3214879", "rt": 0.14}, {"Id": "5182947", "rt": 0.70}, {"Id": "2718594", "rt": 0.91},
#   {"Id": "7826543", "rt": 0.50}, {"Id": "3492857", "rt": 0.48}, {"Id": "2684789", "rt": 0.79}, {"Id": "3491852", "rt": 0.44}, {"Id": "7812459", "rt": 0.53},
#   {"Id": "5692831", "rt": 0.60}, {"Id": "1823467", "rt": 0.31}, {"Id": "3247859", "rt": 0.25}, {"Id": "5496321", "rt": 0.70}, {"Id": "7429815", "rt": 0.84},
#   {"Id": "1857349", "rt": 0.63}, {"Id": "6728493", "rt": 0.26}, {"Id": "2378451", "rt": 0.49}, {"Id": "4817362", "rt": 0.77}, {"Id": "7493812", "rt": 0.12},
#   {"Id": "5987432", "rt": 0.92}, {"Id": "2084759", "rt": 0.15}, {"Id": "1257893", "rt": 0.66}, {"Id": "7481629", "rt": 0.23}, {"Id": "5684231", "rt": 0.10},
#   {"Id": "2138496", "rt": 0.94}, {"Id": "9632584", "rt": 0.41}, {"Id": "4723961", "rt": 0.78}, {"Id": "7834521", "rt": 0.67}, {"Id": "1248693", "rt": 0.33},
#   {"Id": "9127845", "rt": 0.69}, {"Id": "4738216", "rt": 0.82}, {"Id": "3647928", "rt": 0.07}, {"Id": "5784921", "rt": 0.51}, {"Id": "9174825", "rt": 0.25},
#   {"Id": "2894715", "rt": 0.74}, {"Id": "3857429", "rt": 0.03}, {"Id": "4258673", "rt": 0.91}, {"Id": "5714398", "rt": 0.86}, {"Id": "3714259", "rt": 0.19},
#   {"Id": "8624173", "rt": 0.60}, {"Id": "5146287", "rt": 0.21}, {"Id": "6784329", "rt": 0.41}, {"Id": "2147865", "rt": 0.38}, {"Id": "3852164", "rt": 0.85},
#   {"Id": "7954238", "rt": 0.17}, {"Id": "2914876", "rt": 0.76}, {"Id": "6283957", "rt": 0.79}, {"Id": "5314789", "rt": 0.44}, {"Id": "1825947", "rt": 0.34},
#   {"Id": "7324895", "rt": 0.63}, {"Id": "1987354", "rt": 0.98}, {"Id": "6184732", "rt": 0.52}, {"Id": "7184259", "rt": 0.33}, {"Id": "4587129", "rt": 0.64},
#   {"Id": "5714926", "rt": 0.49}, {"Id": "1938746", "rt": 0.25}, {"Id": "9324781", "rt": 0.56}, {"Id": "2497185", "rt": 0.85}, {"Id": "5671843", "rt": 0.17},
#   {"Id": "1237548", "rt": 0.70}, {"Id": "4831752", "rt": 0.73}, {"Id": "7148932", "rt": 0.37}, {"Id": "5871349", "rt": 0.11}, {"Id": "9847213", "rt": 0.59},
#   {"Id": "3187429", "rt": 0.40}, {"Id": "9624715", "rt": 0.21}, {"Id": "5618742", "rt": 0.91}, {"Id": "7421853", "rt": 0.65}, {"Id": "3857194", "rt": 0.34},
#   {"Id": "8914725", "rt": 0.84}, {"Id": "3687425", "rt": 0.02}, {"Id": "4056739", "rt": 0.70}, {"Id": "9123458", "rt": 0.16}, {"Id": "5371489", "rt": 0.28},
#   {"Id": "2348175", "rt": 0.52}, {"Id": "1394728", "rt": 0.93}, {"Id": "2475819", "rt": 0.48}, {"Id": "7418539", "rt": 0.54}, {"Id": "6739241", "rt": 0.49},
#   {"Id": "2984375", "rt": 0.35}, {"Id": "5417692", "rt": 0.61}, {"Id": "3962784", "rt": 0.47}, {"Id": "7583496", "rt": 0.92}, {"Id": "3418752", "rt": 0.39},
#   {"Id": "5869712", "rt": 0.66}, {"Id": "7251938", "rt": 0.83}, {"Id": "9841573", "rt": 0.73}, {"Id": "5628941", "rt": 0.50}, {"Id": "2849571", "rt": 0.28},
#   {"Id": "4198573", "rt": 0.06}, {"Id": "7516329", "rt": 0.55}, {"Id": "8145729", "rt": 0.16}, {"Id": "2459861", "rt": 0.94}, {"Id": "1736852", "rt": 0.78},
#   {"Id": "5692417", "rt": 0.42}, {"Id": "3852614", "rt": 0.67}, {"Id": "7928564", "rt": 0.29}, {"Id": "2719468", "rt": 0.12}, {"Id": "4178356", "rt": 0.33},
#   {"Id": "5386321", "rt": 0.74}, {"Id": "2648195", "rt": 0.21}, {"Id": "7815624", "rt": 0.92}, {"Id": "3498715", "rt": 0.59}, {"Id": "4725869", "rt": 0.62},
#   {"Id": "6128943", "rt": 0.45}, {"Id": "7594812", "rt": 0.93}, {"Id": "3284175", "rt": 0.84}, {"Id": "9147284", "rt": 0.22}, {"Id": "1087524", "rt": 0.79},
#   {"Id": "2837546", "rt": 0.64}, {"Id": "7482139", "rt": 0.63}, {"Id": "5091834", "rt": 0.32}, {"Id": "8713549", "rt": 0.15}, {"Id": "6741532", "rt": 0.66},
#   {"Id": "3594872", "rt": 0.94}, {"Id": "8124569", "rt": 0.48}, {"Id": "4159278", "rt": 0.83}, {"Id": "7538246", "rt": 0.25}, {"Id": "2481576", "rt": 0.52},
#   {"Id": "3195467", "rt": 0.57}, {"Id": "9176438", "rt": 0.29}, {"Id": "7428639", "rt": 0.58}, {"Id": "6234581", "rt": 0.77}, {"Id": "5193274", "rt": 0.91},
#   {"Id": "6912748", "rt": 0.46}, {"Id": "2538417", "rt": 0.82}, {"Id": "8947513", "rt": 0.03}, {"Id": "9286471", "rt": 0.49}, {"Id": "6143728", "rt": 0.94},
#   {"Id": "9176452", "rt": 0.54}, {"Id": "2854691", "rt": 0.68}, {"Id": "1768542", "rt": 0.40}, {"Id": "8421576", "rt": 0.22}, {"Id": "6412573", "rt": 0.51},
#   {"Id": "7318452", "rt": 0.69}, {"Id": "1795426", "rt": 0.10}, {"Id": "2189763", "rt": 0.88}, {"Id": "4978621", "rt": 0.19}, {"Id": "3514782", "rt": 0.62},
#   {"Id": "7583624", "rt": 0.76}, {"Id": "8243195", "rt": 0.34}, {"Id": "9527841", "rt": 0.63}, {"Id": "5834672", "rt": 0.95}, {"Id": "3719482", "rt": 0.20},
#   {"Id": "5148693", "rt": 0.49}, {"Id": "2498573", "rt": 0.71}, {"Id": "6831472", "rt": 0.12}, {"Id": "2315784", "rt": 0.35}, {"Id": "4785163", "rt": 0.66},
#   {"Id": "5724613", "rt": 0.57}, {"Id": "8134927", "rt": 0.38}, {"Id": "6782519", "rt": 0.90}, {"Id": "2958471", "rt": 0.45}, {"Id": "3671594", "rt": 0.15},
#   {"Id": "1245789", "rt": 0.77}, {"Id": "3497281", "rt": 0.56}, {"Id": "5849263", "rt": 0.84}, {"Id": "2761948", "rt": 0.53}, {"Id": "9861724", "rt": 0.0},
# ]
{"Id":"968ABC8","rt":1.0},{"Id":"156ABC8","rt":1.0},{"Id":"392ABC7","rt":1.0},{"Id":"155ABC8","rt":1.0},{"Id":"29ABC02","rt":1.0},{"Id":"805ABC3","rt":1.0},{"Id":"192ABC9","rt":1.0},{"Id":"039ABC7","rt":0.92},{"Id":"375ABC2","rt":1.0},{"Id":"251ABC3","rt":1.0},{"Id":"64ABC03","rt":1.0},{"Id":"662ABC0","rt":1.0},{"Id":"296ABC2","rt":0.89},{"Id":"226ABC3","rt":0.94},{"Id":"442ABC6","rt":0.92},{"Id":"435ABC3","rt":0.86},{"Id":"804ABC2","rt":1.0},{"Id":"013ABC7","rt":1.0},{"Id":"258ABC5","rt":0.88},{"Id":"8ABC007","rt":1.0},{"Id":"319ABC1","rt":0.69},{"Id":"033ABC4","rt":1.0},{"Id":"509ABC1","rt":0.98},{"Id":"48ABC00","rt":0.82},{"Id":"259ABC7","rt":1.0},{"Id":"681ABC8","rt":1.0},{"Id":"29ABC08","rt":1.0},{"Id":"52ABC07","rt":1.0},{"Id":"846ABC4","rt":0.93},{"Id":"06ABC01","rt":1.0},{"Id":"278ABC1","rt":1.0},{"Id":"953ABC3","rt":0.92},{"Id":"037ABC6","rt":0.76},{"Id":"054ABC9","rt":0.99},{"Id":"166ABC1","rt":0.98},{"Id":"977ABC2","rt":0.98},{"Id":"589ABC7","rt":0.85},{"Id":"722ABC0","rt":1.0},{"Id":"167ABC0","rt":1.0},{"Id":"114ABC5","rt":1.0},{"Id":"031ABC5","rt":0.77},{"Id":"819ABC9","rt":0.96},{"Id":"156ABC6","rt":0.79},{"Id":"453ABC3","rt":1.0},{"Id":"262ABC1","rt":1.0},{"Id":"05ABC09","rt":1.0},{"Id":"044ABC4","rt":1.0},{"Id":"892ABC8","rt":1.0},{"Id":"135ABC8","rt":0.96},{"Id":"61ABC00","rt":1.0},{"Id":"142ABC9","rt":0.6},{"Id":"272ABC6","rt":0.87},{"Id":"727ABC2","rt":0.98},{"Id":"411ABC5","rt":1.0},{"Id":"363ABC7","rt":0.86},
   # {"Id":"178ABC5","rt":0.85},{"Id":"47ABC02","rt":0.71},{"Id":"375ABC5","rt":1.0},{"Id":"433ABC2","rt":0.64},
  #  {"Id":"655ABC0","rt":0.88},{"Id":"869ABC3","rt":1.0},{"Id":"663ABC7","rt":1.0},{"Id":"362ABC9","rt":0.97},{"Id":"802ABC3","rt":0.97},{"Id":"39ABC05","rt":0.91},{"Id":"128ABC7","rt":1.0},{"Id":"292ABC0","rt":0.93},{"Id":"029ABC6","rt":1.0},{"Id":"835ABC8","rt":0.89},{"Id":"936ABC0","rt":0.93},{"Id":"038ABC7","rt":1.0},{"Id":"914ABC0","rt":0.91},{"Id":"858ABC1","rt":1.0},{"Id":"529ABC9","rt":1.0},{"Id":"0330029","rt":1.0},{"Id":"78ABC05","rt":0.73},{"Id":"915ABC4","rt":1.0},{"Id":"7ABC009","rt":1.0},{"Id":"753ABC6","rt":1.0},{"Id":"865ABC1","rt":1.0},{"Id":"367ABC7","rt":1.0},{"Id":"103ABC1","rt":1.0},{"Id":"001ABC7","rt":0.66},{"Id":"739ABC8","rt":1.0},{"Id":"02ABC08","rt":1.0},{"Id":"657ABC4","rt":1.0},{"Id":"211ABC9","rt":1.0},{"Id":"543ABC2","rt":0.87},{"Id":"034ABC9","rt":0.76},{"Id":"89ABC06","rt":0.92},{"Id":"159ABC1","rt":0.97},{"Id":"81ABC04","rt":1.0},{"Id":"507ABC9","rt":1.0},{"Id":"677ABC7","rt":0.95},{"Id":"389ABC4","rt":0.93},{"Id":"089ABC8","rt":1.0},{"Id":"045ABC7","rt":0.83},{"Id":"965ABC0","rt":1.0},{"Id":"864ABC0","rt":0.43},{"Id":"993ABC0","rt":1.0},{"Id":"246ABC7","rt":1.0},{"Id":"401ABC5","rt":1.0},{"Id":"261ABC6","rt":0.57},{"Id":"296ABC1","rt":1.0},{"Id":"015ABC3","rt":1.0},{"Id":"414ABC9","rt":1.0},{"Id":"164ABC9","rt":1.0},{"Id":"646ABC2","rt":1.0},{"Id":"385ABC2","rt":1.0},{"Id":"038ABC6","rt":1.0},{"Id":"508ABC3","rt":1.0},{"Id":"7310047","rt":1.0},{"Id":"999ABC4","rt":1.0},{"Id":"518ABC6","rt":0.92},{"Id":"364ABC4","rt":1.0},{"Id":"215ABC7","rt":0.99},{"Id":"41ABC01","rt":0.99},{"Id":"205ABC2","rt":0.99},{"Id":"271ABC9","rt":0.96},{"Id":"165ABC8","rt":0.92},{"Id":"456ABC7","rt":0.97},{"Id":"345ABC4","rt":0.93},{"Id":"056ABC5","rt":0.96},{"Id":"13ABC09","rt":0.96},{"Id":"46ABC06","rt":0.93},{"Id":"933ABC2","rt":0.96},{"Id":"339ABC5","rt":0.88},{"Id":"47ABC00","rt":0.74},{"Id":"004ABC0","rt":0.79},{"Id":"082ABC5","rt":0.95},{"Id":"894ABC2","rt":0.95},{"Id":"658ABC2","rt":0.93},{"Id":"644ABC4","rt":0.89},{"Id":"105ABC5","rt":0.95},{"Id":"704ABC9","rt":0.93},{"Id":"349ABC4","rt":0.93},{"Id":"647ABC6","rt":0.94},{"Id":"939ABC5","rt":0.93},{"Id":"196ABC3","rt":0.87},{"Id":"088ABC1","rt":0.93},{"Id":"569ABC4","rt":0.93},{"Id":"454ABC9","rt":0.93},{"Id":"871ABC6","rt":0.93},{"Id":"123ABC3","rt":0.92},{"Id":"888ABC0","rt":0.92},{"Id":"323ABC4","rt":0.89},{"Id":"09ABC03","rt":0.88},{"Id":"943ABC0","rt":0.91},{"Id":"866ABC5","rt":0.91},{"Id":"086ABC3","rt":0.91},{"Id":"52ABC01","rt":0.86},{"Id":"897ABC1","rt":0.81},{"Id":"451ABC2","rt":0.91},{"Id":"532ABC9","rt":0.89},{"Id":"849ABC6","rt":0.9},{"Id":"463ABC7","rt":0.91},{"Id":"648ABC8","rt":0.9},{"Id":"493ABC3","rt":0.79},{"Id":"542ABC0","rt":0.89},{"Id":"309ABC1","rt":0.89},{"Id":"66ABC01","rt":0.9},{"Id":"029ABC4","rt":0.86},{"Id":"409ABC7","rt":0.89},{"Id":"929ABC9","rt":0.85},{"Id":"162ABC3","rt":0.87},{"Id":"471ABC6","rt":0.89},{"Id":"184ABC7","rt":0.89},{"Id":"945ABC6","rt":0.88},{"Id":"286ABC2","rt":0.88},{"Id":"258ABC1","rt":0.88},{"Id":"685ABC3","rt":0.88},{"Id":"874ABC9","rt":0.88},{"Id":"084ABC4","rt":0.88},{"Id":"112ABC3","rt":0.88},{"Id":"882ABC4","rt":0.88},{"Id":"42ABC08","rt":0.88},{"Id":"022ABC4","rt":0.84},{"Id":"709ABC6","rt":0.79},{"Id":"093ABC0","rt":0.87},{"Id":"683ABC6","rt":0.87},{"Id":"059ABC3","rt":0.78},{"Id":"601ABC9","rt":0.86},{"Id":"026ABC4","rt":0.86},{"Id":"893ABC8","rt":0.86},{"Id":"041ABC8","rt":0.86},{"Id":"916ABC1","rt":0.82},{"Id":"394ABC7","rt":0.86},{"Id":"434ABC2","rt":0.86},{"Id":"505ABC9","rt":0.86},{"Id":"293ABC7","rt":0.85},{"Id":"217ABC6","rt":0.82},{"Id":"906ABC1","rt":0.7},{"Id":"385ABC6","rt":0.85},{"Id":"876ABC7","rt":0.81},{"Id":"753ABC4","rt":0.82},{"Id":"025ABC4","rt":0.84},{"Id":"701ABC1","rt":0.79},{"Id":"661ABC4","rt":0.84},{"Id":"37ABC09","rt":0.84},{"Id":"728ABC4","rt":0.84},{"Id":"141ABC7","rt":0.84},{"Id":"807ABC5","rt":0.84},{"Id":"329ABC0","rt":0.74},{"Id":"929ABC5","rt":0.83},{"Id":"914ABC5","rt":0.83},{"Id":"823ABC9","rt":0.83},{"Id":"834ABC5","rt":0.83},{"Id":"738ABC2","rt":0.76},{"Id":"941ABC0","rt":0.8},{"Id":"886ABC2","rt":0.74},{"Id":"387ABC7","rt":0.78},{"Id":"902ABC0","rt":0.8},{"Id":"877ABC4","rt":0.82},{"Id":"298ABC6","rt":0.82},{"Id":"928ABC0","rt":0.81},{"Id":"329ABC0","rt":0.8},{"Id":"899ABC2","rt":0.81},{"Id":"649ABC0","rt":0.81},{"Id":"882ABC6","rt":0.81},{"Id":"826ABC2","rt":0.81},{"Id":"162ABC4","rt":0.79},{"Id":"077ABC4","rt":0.68},{"Id":"626ABC9","rt":0.79},{"Id":"891ABC9","rt":0.78},{"Id":"192ABC7","rt":0.79},{"Id":"684ABC8","rt":0.8},{"Id":"599ABC0","rt":0.79},{"Id":"191ABC4","rt":0.65},{"Id":"629ABC6","rt":0.79},{"Id":"527ABC0","rt":0.79},{"Id":"098ABC8","rt":0.77},{"Id":"562ABC2","rt":0.79},{"Id":"152ABC5","rt":0.76},{"Id":"208ABC7","rt":0.78},{"Id":"51ABC06","rt":0.59},{"Id":"49ABC01","rt":0.78},{"Id":"88ABC07","rt":0.78},{"Id":"514ABC8","rt":0.78},{"Id":"32ABC06","rt":0.78},{"Id":"693ABC2","rt":0.72},{"Id":"823ABC4","rt":0.77},{"Id":"136ABC6","rt":0.78},{"Id":"95ABC00","rt":0.78},{"Id":"912ABC2","rt":0.77},{"Id":"369ABC1","rt":0.77},{"Id":"353ABC1","rt":0.77},{"Id":"923ABC8","rt":0.59},{"Id":"721ABC0","rt":0.76},{"Id":"364ABC1","rt":0.76},{"Id":"756ABC2","rt":0.76},{"Id":"588ABC8","rt":0.75},{"Id":"624ABC2","rt":0.76},{"Id":"019ABC8","rt":0.76},{"Id":"708ABC8","rt":0.76},{"Id":"322ABC0","rt":0.76},{"Id":"107ABC8","rt":0.75},{"Id":"738ABC7","rt":0.75},{"Id":"879ABC0","rt":0.74},{"Id":"597ABC1","rt":0.75},{"Id":"745ABC4","rt":0.75},{"Id":"758ABC8","rt":0.75},{"Id":"213ABC7","rt":0.74},{"Id":"671ABC0","rt":0.71},{"Id":"775ABC9","rt":0.69},{"Id":"782ABC4","rt":0.74},{"Id":"536ABC0","rt":0.72},{"Id":"482ABC5","rt":0.74},{"Id":"998ABC2","rt":0.71},{"Id":"389ABC0","rt":0.55},{"Id":"3ABC005","rt":0.73},{"Id":"244ABC8","rt":0.68},{"Id":"585ABC8","rt":0.73},{"Id":"924ABC4","rt":0.73},{"Id":"199ABC1","rt":0.73},{"Id":"917ABC0","rt":0.73},{"Id":"794ABC6","rt":0.72},{"Id":"421ABC3","rt":0.73},{"Id":"744ABC5","rt":0.63},{"Id":"121ABC2","rt":0.72},{"Id":"253ABC5","rt":0.72},{"Id":"375ABC8","rt":0.7},{"Id":"827ABC0","rt":0.72},{"Id":"508ABC6","rt":0.7},{"Id":"585ABC3","rt":0.71},{"Id":"385ABC3","rt":0.71},{"Id":"796ABC2","rt":0.71},{"Id":"682ABC2","rt":0.71},{"Id":"089ABC1","rt":0.7},{"Id":"081ABC1","rt":0.7},{"Id":"892ABC6","rt":0.7},{"Id":"324ABC0","rt":0.69},{"Id":"558ABC9","rt":0.69},{"Id":"487ABC0","rt":0.4},{"Id":"403ABC9","rt":0.64},{"Id":"28ABC05","rt":0.64},{"Id":"939ABC7","rt":0.67},{"Id":"021ABC7","rt":0.68},{"Id":"011ABC5","rt":0.68},{"Id":"32ABC04","rt":0.68},{"Id":"285ABC7","rt":0.61},{"Id":"993ABC4","rt":0.67},{"Id":"092ABC8","rt":0.67},{"Id":"216ABC9","rt":0.67},{"Id":"388ABC9","rt":0.67},{"Id":"296ABC5","rt":0.67},{"Id":"779ABC6","rt":0.66},{"Id":"895ABC0","rt":0.66},{"Id":"934ABC1","rt":0.64},{"Id":"438ABC6","rt":0.52},{"Id":"771ABC5","rt":0.61},{"Id":"596ABC4","rt":0.51},{"Id":"352ABC6","rt":0.64},{"Id":"64ABC09","rt":0.64},{"Id":"486ABC8","rt":0.64},{"Id":"693ABC9","rt":0.64},{"Id":"294ABC4","rt":0.64},{"Id":"052ABC3","rt":0.63},{"Id":"402ABC3","rt":0.63},{"Id":"102ABC5","rt":0.62},{"Id":"601ABC0","rt":0.62},{"Id":"896ABC6","rt":0.62},{"Id":"039ABC4","rt":0.62},{"Id":"577ABC8","rt":0.6},{"Id":"353ABC4","rt":0.61},{"Id":"01ABC06","rt":0.6},{"Id":"818ABC3","rt":0.59},{"Id":"455ABC5","rt":0.59},{"Id":"768ABC5","rt":0.59},{"Id":"184ABC3","rt":0.53},{"Id":"106ABC2","rt":0.58},{"Id":"401ABC4","rt":0.58},{"Id":"337ABC7","rt":0.58},{"Id":"37ABC02","rt":0.57},{"Id":"697ABC1","rt":0.48},{"Id":"384ABC3","rt":0.56},{"Id":"11ABC08","rt":0.55},{"Id":"735ABC2","rt":0.49},{"Id":"657ABC5","rt":0.46},{"Id":"074ABC5","rt":0.49},{"Id":"335ABC1","rt":0.54},{"Id":"032ABC9","rt":0.54},{"Id":"671ABC3","rt":0.54},{"Id":"148ABC0","rt":0.53},{"Id":"082ABC0","rt":0.52},{"Id":"748ABC3","rt":0.51},{"Id":"48ABC01","rt":0.51},{"Id":"749ABC9","rt":0.48},{"Id":"071ABC1","rt":0.5},{"Id":"679ABC6","rt":0.5},{"Id":"85ABC04","rt":0.49},{"Id":"663ABC5","rt":0.45},{"Id":"944ABC9","rt":0.36},{"Id":"111ABC5","rt":0.41},{"Id":"854ABC2","rt":0.48},{"Id":"885ABC2","rt":0.46},{"Id":"543ABC7","rt":0.45},{"Id":"681ABC1","rt":0.46},{"Id":"113ABC4","rt":0.44},
   {"Id":"27ABC07","rt":0.37},{"Id":"666ABC5","rt":0.41},{"Id":"852ABC6","rt":0.39},{"Id":"911ABC8","rt":0.37},{"Id":"901ABC6","rt":0.35},{"Id":"279ABC4","rt":0.28},{"Id":"274ABC1","rt":0.29},{"Id":"113ABC5","rt":0.25},{"Id":"521ABC6","rt":0.25},{"Id":"084ABC0","rt":0.25},{"Id":"007ABC4","rt":0.22},{"Id":"724ABC9","rt":0.11},{"Id":"821ABC9","rt":0.0},{"Id":"05ABC01","rt":0.0},{"Id":"054ABC9","rt":0.0},{"Id":"695ABC2","rt":0.0},{"Id":"661ABC6","rt":0.0},{"Id":"713ABC9","rt":0.0},{"Id":"157ABC4","rt":0.0},{"Id":"332ABC7","rt":0.0},{"Id":"544ABC6","rt":0.0},{"Id":"924ABC5","rt":0.0},{"Id":"066ABC9","rt":0.0},{"Id":"482ABC0","rt":0.0},{"Id":"438ABC3","rt":0.0},{"Id":"482ABC6","rt":0.0},{"Id":"483ABC9","rt":0.0},{"Id":"617ABC4","rt":0.0},{"Id":"35ABC05","rt":0.0},{"Id":"447ABC2","rt":0.0},{"Id":"085ABC2","rt":0.0},{"Id":"886ABC8","rt":0.0},{"Id":"546ABC3","rt":0.0},{"Id":"813ABC1","rt":0.0},{"Id":"53ABC04","rt":0.0},{"Id":"922ABC9","rt":0.0},{"Id":"351ABC3","rt":0.0},{"Id":"208ABC4","rt":0.0},{"Id":"431ABC5","rt":0.0},{"Id":"113ABC6","rt":0.0},{"Id":"086ABC1","rt":0.0},{"Id":"067ABC3","rt":0.0},{"Id":"611ABC4","rt":0.0},{"Id":"192ABC7","rt":0.0},{"Id":"82ABC00","rt":0.0},{"Id":"865ABC9","rt":0.0},{"Id":"977ABC1","rt":0.0},{"Id":"23ABC05","rt":0.0},{"Id":"958ABC2","rt":0.0},{"Id":"197ABC4","rt":0.0},{"Id":"508ABC1","rt":0.0},{"Id":"398ABC5","rt":0.0},{"Id":"957ABC4","rt":0.0},{"Id":"719ABC7","rt":0.0},{"Id":"886ABC7","rt":0.0},{"Id":"715ABC0","rt":0.0},{"Id":"699ABC8","rt":0.0},{"Id":"229ABC7","rt":0.0},{"Id":"268ABC0","rt":0.0},{"Id":"341ABC4","rt":0.0},{"Id":"848ABC4","rt":0.0},{"Id":"ABCABC7","rt":0.0},{"Id":"673ABC5","rt":0.0},{"Id":"823ABC2","rt":0.0},{"Id":"567ABC6","rt":0.0},{"Id":"553ABC0","rt":0.0},{"Id":"27ABC00","rt":0.0},{"Id":"73ABC00","rt":0.0},{"Id":"656ABC9","rt":0.0},{"Id":"787ABC1","rt":0.0},{"Id":"511ABC7","rt":0.0},{"Id":"526ABC1","rt":0.0},{"Id":"357ABC0","rt":0.0},{"Id":"278ABC7","rt":0.0},{"Id":"729ABC0","rt":0.0},{"Id":"829ABC9","rt":0.0},{"Id":"227ABC6","rt":0.0},{"Id":"562ABC6","rt":0.0},{"Id":"502ABC1","rt":0.0},{"Id":"615ABC9","rt":0.0},{"Id":"688ABC5","rt":0.0},{"Id":"779ABC2","rt":0.0},{"Id":"ABCABC3","rt":0.0},
]



# Step 3. Define the state
class AgentState(TypedDict):
    sellers: List[Dict[str, float]]
    output: Optional[List[Dict[str, float]]]


# Step 4. Define graph nodes
def generate_factors(state: AgentState) -> AgentState:
    """Use the LLM to map seller rates to factors."""
    print("Generating factors with LLM...")

    sellers = state.get("sellers", [])

    # Build prompt with anchor mapping instructions
    prompt = f"""
You are given a list of records with IDs and rt (rate between 0 and 1).
Map each rate to a factor using the following nonlinear anchor points:

rate 1.0 -> factor 15
rate 0.88 -> factor 28
rate 0.80 -> factor 35
rate 0.74 -> factor 40.5
rate 0.72 -> factor 42
rate 0.0 -> factor 55

Important:
- Do NOT use linear interpolation. Assume a curved mapping between anchors.
- Min value is 15. Max value is 55. The generated factor cannot be less than 15 or greater than 55.
- Average factor should be 35.
- Return ONLY valid JSON in this exact format (no markdown, no code blocks, no explanation):
{{"output":[{{"Id":"1234567","rt":0.88,"factor":28}}, ...]}}

Here is the input list:
{sellers}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()

    # Extract JSON if wrapped in code fences
    if "```" in response_text:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
    elif response_text.startswith("{") and response_text.endswith("}"):
        pass
    else:
        json_match = re.search(r'\{[^{}]*"output"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)

    print(f"LLM response: {response_text}")

    try:
        structured = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise ValueError(f"Failed to parse LLM response as JSON: {response_text}")

    return {**state, "output": structured.get("output", [])}

def generate_distribution_chart(output_data: List[Dict[str, float]], save_path: str = "rate_factor_distribution.png"):
    """Generate a distribution chart showing rate vs factor relationship."""
    if not output_data:
        print("No data available for chart generation")
        return

    # Extract rates and factors using correct keys
    rates = [item.get("rt", 0) for item in output_data]
    factors = [item.get("factor", 0) for item in output_data]
    seller_ids = [item.get("Id", "") for item in output_data]

    # Filter out any zero or invalid data
    valid_data = [(r, f, sid) for r, f, sid in zip(rates, factors, seller_ids) if f > 0]
    if not valid_data:
        print("No valid data points for chart generation")
        return

    rates, factors, seller_ids = zip(*valid_data)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Create scatter plot - swapped axes: factors on X, rates on Y
    plt.scatter(factors, rates, s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1)

    # Add labels for each point (only show every 10th label to avoid overcrowding)
    for i, seller_id in enumerate(seller_ids):
        if i % 10 == 0 or len(seller_ids) < 20:  # Show fewer labels if many points
            plt.annotate(f'ID: {seller_id}', 
                        (factors[i], rates[i]), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.6)

    # Customize the plot - swapped labels
    plt.xlabel('Factor', fontsize=14, fontweight='bold')
    plt.ylabel('Rate (rt)', fontsize=14, fontweight='bold')
    plt.title('Seller Factor vs Rate Distribution', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Set axis limits with some padding, handle edge cases - swapped axes
    if rates and factors:
        rate_range = max(rates) - min(rates)
        factor_range = max(factors) - min(factors)

        # Handle case where range is zero or very small
        if factor_range < 1e-10:
            factor_padding = 2.0
        else:
            factor_padding = 0.05 * factor_range

        if rate_range < 1e-10:
            rate_padding = 0.1
        else:
            rate_padding = 0.05 * rate_range

        plt.xlim(min(factors) - factor_padding, max(factors) + factor_padding)
        plt.ylim(min(rates) - rate_padding, max(rates) + rate_padding)

    # Add trend line if we have enough data points and variance - swapped for new axes
    if len(rates) > 1:
        try:
            # Check if there's enough variance in the data
            rate_std = np.std(rates)
            factor_std = np.std(factors)

            if rate_std > 1e-10 and factor_std > 1e-10:
                z = np.polyfit(factors, rates, 1)  # factors as X, rates as Y
                p = np.poly1d(z)
                x_trend = np.linspace(min(factors), max(factors), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                        label=f'Trend Line (slope: {z[0]:.3f})')
                plt.legend()
            else:
                print("Data has insufficient variance for trend line calculation")
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Could not calculate trend line: {e}")

    # Add summary statistics
    plt.figtext(0.02, 0.02, f"Data Points: {len(rates)}", fontsize=10, alpha=0.7)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {save_path}")

    # Show the plot
    plt.show()


def generate_standard_deviation_chart(output_data: List[Dict[str, float]], save_path: str = "standard_deviation_chart.png"):
    """Generate a standard deviation chart showing the distribution of factors with std dev bands."""
    if not output_data:
        print("No data available for standard deviation chart generation")
        return

    # Extract factors using correct keys
    factors = [item.get("factor", 0) for item in output_data if item.get("factor", 0) > 0]
    
    if not factors or len(factors) < 2:
        print("Insufficient data for standard deviation chart generation")
        return

    # Calculate statistics
    mean_factor = np.mean(factors)
    std_factor = np.std(factors)
    median_factor = np.median(factors)
    min_factor = np.min(factors)
    max_factor = np.max(factors)

    print(f"\nStandard Deviation Statistics:")
    print(f"Mean: {mean_factor:.2f}")
    print(f"Standard Deviation: {std_factor:.2f}")
    print(f"Median: {median_factor:.2f}")
    print(f"Min: {min_factor:.2f}")
    print(f"Max: {max_factor:.2f}")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Subplot 1: Histogram with standard deviation bands
    n_bins = min(30, int(np.sqrt(len(factors))))  # Adaptive bin count
    counts, bins, patches = ax1.hist(factors, bins=n_bins, edgecolor='black', alpha=0.7, 
                                     color='steelblue', label='Factor Distribution')
    
    # Color bars based on standard deviation zones
    for i, (count, bin_left, bin_right, patch) in enumerate(zip(counts, bins[:-1], bins[1:], patches)):
        bin_center = (bin_left + bin_right) / 2
        if abs(bin_center - mean_factor) <= std_factor:
            patch.set_facecolor('lightgreen')  # Within 1σ
        elif abs(bin_center - mean_factor) <= 2 * std_factor:
            patch.set_facecolor('yellow')  # Within 2σ
        elif abs(bin_center - mean_factor) <= 3 * std_factor:
            patch.set_facecolor('orange')  # Within 3σ
        else:
            patch.set_facecolor('red')  # Beyond 3σ

    # Draw vertical lines for mean and standard deviation bands
    ax1.axvline(mean_factor, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_factor:.2f}')
    ax1.axvline(mean_factor + std_factor, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'±1σ: {std_factor:.2f}')
    ax1.axvline(mean_factor - std_factor, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(mean_factor + 2 * std_factor, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'±2σ: {2*std_factor:.2f}')
    ax1.axvline(mean_factor - 2 * std_factor, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(mean_factor + 3 * std_factor, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label=f'±3σ: {3*std_factor:.2f}')
    ax1.axvline(mean_factor - 3 * std_factor, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)

    # Fill areas for standard deviation zones
    ax1.axvspan(mean_factor - std_factor, mean_factor + std_factor, alpha=0.1, color='green', label='±1σ Zone (68%)')
    ax1.axvspan(mean_factor - 2 * std_factor, mean_factor + 2 * std_factor, alpha=0.1, color='orange', label='±2σ Zone (95%)')
    ax1.axvspan(mean_factor - 3 * std_factor, mean_factor + 3 * std_factor, alpha=0.1, color='purple', label='±3σ Zone (99.7%)')

    ax1.set_xlabel('Factor Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Factor Distribution with Standard Deviation Bands', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # Subplot 2: Box plot with standard deviation annotations
    bp = ax2.boxplot(factors, vert=True, patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5))
    
    # Add standard deviation lines
    ax2.axhline(mean_factor, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_factor:.2f}')
    ax2.axhline(mean_factor + std_factor, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(mean_factor - std_factor, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'±1σ: {std_factor:.2f}')
    ax2.axhline(mean_factor + 2 * std_factor, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(mean_factor - 2 * std_factor, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'±2σ: {2*std_factor:.2f}')
    ax2.axhline(mean_factor + 3 * std_factor, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(mean_factor - 3 * std_factor, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label=f'±3σ: {3*std_factor:.2f}')

    ax2.set_ylabel('Factor Value', fontsize=12, fontweight='bold')
    ax2.set_title('Factor Box Plot with Standard Deviation Reference Lines', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xticklabels(['Factors'])

    # Add statistics text box
    stats_text = f"""Statistics:
    Count: {len(factors)}
    Mean: {mean_factor:.2f}
    Median: {median_factor:.2f}
    Std Dev: {std_factor:.2f}
    Min: {min_factor:.2f}
    Max: {max_factor:.2f}
    Range: {max_factor - min_factor:.2f}
    CV: {(std_factor/mean_factor)*100:.2f}%"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Standard deviation chart saved as: {save_path}")

    # Show the plot
    plt.show()


# Step 5. Build graph
graph = StateGraph(AgentState)
graph.add_node("generate_factors", generate_factors)
graph.add_edge(START, "generate_factors")
graph.add_edge("generate_factors", END)

# Step 6. Compile
agent = graph.compile()

# Step 7. Run example
if __name__ == "__main__":
    result = agent.invoke({"sellers": mocked_sellers})
    print("Final Output:")
    print(json.dumps(result["output"], indent=2))

    # Generate and display distribution chart
    if result.get("output"):
        print("\nGenerating distribution chart...")
        generate_distribution_chart(result["output"])
        
        print("\nGenerating standard deviation chart...")
        generate_standard_deviation_chart(result["output"])
    else:
        print("No output data available for chart generation")
