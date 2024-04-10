from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annFile = './xtrt/data/coco/annotations/instances_val2017.json'  
resFile = 'results.json'  

cocoGt = COCO(annFile)
cocoDt = cocoGt.loadRes(resFile)
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
