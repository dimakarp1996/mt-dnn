import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
tasks=['topics','datopics','daintents','emo','sentiment','toxic','factoid'][::-1]
checkpoint_dir='/cephfs/home/karpov.d/mt-dnn/checkpoints/dream_adamax_answer_opt1_gc0_ggc1_2021-11-09T1439'
for task in tasks:
    json_file = task+'_test_scores_epoch_3.json'
    pred = json.load(open(checkpoint_dir+'/'+json_file, 'r'))['predictions']
    scores = [json.loads(k)['label'] for k in open(f'experiments/dream/canonical_data/bert-base-uncased/{task}_test.json','r').readlines()]
    print(task)
    print(accuracy_score(scores,pred))
    print(f1_score(scores,pred,average='macro'))
    if task=='toxic':
        print(scores[:30])
        print(pred[:30])
        scores = [s==7 for s in scores]
        pred = [s==7 for s in pred]

        print('F1 tox/nontox')
        print(classification_report(scores,pred))
    elif task=='sentiment':
        print(classification_report(scores,pred))


