'''

Data for use in pgm_tutorial.py

@author: miker@kth.se
@date: 2017-08-30

'''

RAW_DATA = {
    'age': ['>23', '>23', '>23', '>23', '>23', '>23', '>23', '>23', '>23', '>23',
         '20-23', '20-23', '>23', '20-23', '>23', '20-23', '>23', '20-23',
         '20-23', '20-23', '>23', '20-23', '20-23', '20-23', '20-23', '20-23',
         '20-23', '20-23', '20-23', '20-23', '20-23', '20-23', '20-23', '20-23',
         '20-23', '20-23', '20-23', '20-23', '20-23', '20-23', '<=20', '20-23',
         '20-23', '20-23', '20-23', '<=20', '<=20', '20-23', '20-23', '20-23',
         '20-23', '20-23', '<=20', '20-23', '20-23', '<=20', '<=20', '20-23',
         '<=20', '<=20', '<=20', '20-23', '20-23', '20-23', '20-23', '20-23',
         '20-23', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '20-23',
         '<=20', '<=20', '<=20', '<=20', '20-23', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '20-23', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '20-23', '<=20', '20-23', '<=20', '<=20',
         '<=20', '<=20', '20-23', '<=20', '20-23', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '20-23', '20-23',
         '<=20', '<=20', '20-23', '<=20', '<=20', '<=20', '<=20', '<=20',
         '20-23', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20', '<=20',
         '<=20'],
 'avg_cs': ['3<4', '<2', '3<4', '<2', '3<4', '3<4', '3<4', '4<5', '3<4', '3<4',
            '3<4', '3<4', '<2', '<2', '<2', '3<4', '3<4', '<2', '4<5', '3<4',
            '3<4', '<2', '4<5', '3<4', '3<4', '3<4', '3<4', '<2', '3<4', '<2',
            '<2', '3<4', '3<4', '3<4', '3<4', '3<4', '3<4', '3<4', '2<3', '4<5',
            '3<4', '<2', '4<5', '4<5', '3<4', '3<4', '3<4', '4<5', '3<4', '3<4',
            '4<5', '3<4', '4<5', '<2', '3<4', '<2', '3<4', '<2', '3<4', '4<5',
            '3<4', '3<4', '<2', '3<4', '3<4', '4<5', '3<4', '3<4', '<2', '3<4',
            '4<5', '4<5', '3<4', '3<4', '4<5', '3<4', '3<4', '<2', '3<4', '3<4',
            '4<5', '3<4', '4<5', '4<5', '4<5', '3<4', '3<4', '3<4', '4<5',
            '3<4', '3<4', '3<4', '3<4', '3<4', '4<5', '3<4', '3<4', '3<4',
            '3<4', '3<4', '<2', '4<5', '3<4', '<2', '4<5', '3<4', '3<4', '4<5',
            '4<5', '3<4', '3<4', '4<5', '4<5', '3<4', '3<4', '4<5', '3<4',
            '3<4', '3<4', '3<4', '<2', '3<4', '<2', '4<5', '3<4', '3<4', '3<4',
            '3<4', '4<5', '4<5', '3<4', '3<4', '3<4', '3<4', '<2', '4<5', '4<5',
            '4<5', '<2', '3<4', '3<4', '3<4', '3<4', '<2', '4<5', '3<4', '4<5',
            '3<4', '4<5', '3<4', '3<4', '4<5', '4<5', '4<5', '3<4', '3<4',
            '4<5', '4<5', '3<4', '3<4', '3<4', '4<5', '4<5', '4<5', '3<4',
            '4<5', '4<5', '4<5', '4<5', '3<4', '3<4', '3<4', '4<5', '4<5',
            '3<4', '4<5', '4<5', '4<5', '4<5', '3<4', '4<5', '4<5', '3<4',
            '3<4', '<2', '4<5', '3<4', '3<4', '3<4', '3<4', '4<5', '4<5', '3<4',
            '<2', '4<5', '3<4', '3<4', '3<4', '4<5', '3<4', '4<5', '3<4', '4<5',
            '4<5', '4<5', '4<5', '4<5', '<2', '<2', '4<5', '3<4', '4<5', '4<5',
            '3<4', '3<4', '4<5', '4<5', '3<4', '3<4', '3<4', '3<4', '4<5',
            '4<5', '3<4', '4<5', '3<4', '4<5', '3<4', '4<5', '4<5', '3<4',
            '3<4', '4<5', '<2', '4<5', '3<4', '<2', '4<5', '4<5', '4<5', '3<4',
            '4<5', '4<5', '<2', '3<4', '3<4', '4<5', '3<4', '3<4', '<2', '3<4',
            '3<4', '4<5', '3<4', '4<5', '4<5', '4<5', '3<4', '3<4', '4<5',
            '3<4', '3<4', '4<5', '4<5', '4<5'],
 'avg_mat': ['2<3', '3<4', '2<3', '<2', '<2', '2<3', '2<3', '3<4', '3<4', '3<4',
             '3<4', '2<3', '<2', '2<3', '<2', '<2', '2<3', '2<3', '3<4', '2<3',
             '2<3', '<2', '3<4', '2<3', '2<3', '3<4', '2<3', '<2', '2<3', '3<4',
             '<2', '3<4', '2<3', '2<3', '2<3', '2<3', '4<5', '2<3', '2<3',
             '3<4', '2<3', '<2', '3<4', '3<4', '2<3', '3<4', '3<4', '2<3',
             '2<3', '2<3', '4<5', '2<3', '3<4', '<2', '2<3', '<2', '2<3', '<2',
             '2<3', '2<3', '3<4', '3<4', '<2', '2<3', '2<3', '3<4', '2<3', '<2',
             '<2', '2<3', '3<4', '3<4', '2<3', '3<4', '2<3', '2<3', '3<4',
             '2<3', '3<4', '2<3', '3<4', '2<3', '3<4', '4<5', '3<4', '2<3',
             '3<4', '2<3', '2<3', '2<3', '3<4', '2<3', '3<4', '2<3', '3<4',
             '2<3', '2<3', '2<3', '3<4', '3<4', '2<3', '4<5', '2<3', '2<3',
             '2<3', '3<4', '2<3', '2<3', '4<5', '2<3', '<2', '2<3', '3<4', '<2',
             '3<4', '3<4', '2<3', '<2', '2<3', '2<3', '3<4', '2<3', '<2', '2<3',
             '<2', '3<4', '3<4', '2<3', '3<4', '3<4', '2<3', '3<4', '2<3',
             '2<3', '<2', '2<3', '4<5', '3<4', '2<3', '2<3', '2<3', '3<4',
             '2<3', '<2', '3<4', '2<3', '4<5', '2<3', '3<4', '3<4', '3<4',
             '4<5', '3<4', '3<4', '2<3', '<2', '2<3', '2<3', '2<3', '2<3',
             '3<4', '2<3', '2<3', '3<4', '2<3', '3<4', '3<4', '4<5', '3<4',
             '2<3', '2<3', '<2', '3<4', '3<4', '2<3', '3<4', '3<4', '4<5',
             '3<4', '2<3', '4<5', '3<4', '2<3', '3<4', '<2', '3<4', '<2', '2<3',
             '2<3', '2<3', '3<4', '3<4', '3<4', '<2', '3<4', '<2', '<2', '3<4',
             '4<5', '2<3', '2<3', '3<4', '3<4', '3<4', '3<4', '<2', '2<3', '<2',
             '<2', '3<4', '3<4', '4<5', '3<4', '4<5', '3<4', '3<4', '3<4', '<2',
             '3<4', '2<3', '2<3', '3<4', '4<5', '3<4', '3<4', '2<3', '3<4',
             '3<4', '3<4', '3<4', '2<3', '3<4', '3<4', '<2', '3<4', '3<4', '<2',
             '<2', '2<3', '3<4', '4<5', '4<5', '3<4', '2<3', '3<4', '2<3',
             '4<5', '3<4', '2<3', '<2', '<2', '2<3', '2<3', '2<3', '4<5', '3<4',
             '4<5', '3<4', '<2', '4<5', '2<3', '2<3', '4<5', '4<5', '4<5'],
 'delay': ['1', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', 'NA',
           '>=2', '0', '>=2', '0', '0', '0', '0', '1', '>=2', '0', '>=2', '0',
           '0', '0', '1', '0', '0', 'NA', '0', '0', '0', '1', '0', '0', '0',
           '0', '0', '0', 'NA', '0', '0', '0', '0', '0', '0', '0', '0', '0',
           '0', '0', 'NA', '>=2', 'NA', '0', 'NA', '>=2', '1', '0', '0', 'NA',
           '0', '0', 'NA', '1', '0', 'NA', '0', '0', '0', '0', '0', '0', '0',
           '0', '0', '0', '0', '0', '0', '0', 'NA', '0', '0', '0', '1', '0',
           '0', '0', '0', '0', '0', '0', '0', '0', '>=2', '0', '0', '0', '0',
           '0', '1', '0', '1', '0', '0', '0', '1', '>=2', 'NA', '0', '>=2', '0',
           '0', '0', '>=2', '0', '1', 'NA', '0', '>=2', '0', '0', '0', '0', '0',
           '1', '1', '0', '0', '0', '0', '>=2', '0', '0', '0', '0', '0', '>=2',
           '0', '0', '>=2', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0',
           '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0',
           '0', '1', '0', '0', '0', '0', '>=2', '0', '0', '0', '0', '>=2', '0',
           '0', '0', '0', '1', '1', '0', '1', '0', '0', '0', '0', '0', '1', '0',
           '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '1',
           'NA', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '1', '0',
           '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0',
           '0', 'NA', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
           '0', 'NA', '>=2', '0', '0', '0', '0', '0', '0', '0', 'NA', '0', '0',
           '0', '0', '0', '0'],
    'gender': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1',
            '1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0',
            '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1',
            '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '0', '1', '1',
            '0', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0',
            '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '1', '1', '1', '0', '1', '1', '1', '1', '0', '0', '1', '1',
            '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '0',
            '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1',
            '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
            '1', '1', '1', '1', '1']
}

def get_random_partition(ratio, seed = None):
    ''' Returns a random partition of RAW_DATA into two disjoint subsets
        containing ratio and (1 - ratio) amount of the samples '''
    import random
    if seed: random.seed(seed)
    size = len(RAW_DATA['age'])
    i_list = list(range(size))
    random.shuffle(i_list)
    first_size = round(ratio*size)
    first_set = {feature: [RAW_DATA[feature][i] for i in i_list[:first_size]] \
                 for feature in RAW_DATA}
    second_set = {feature: [RAW_DATA[feature][i] for i in i_list[first_size:]] \
                 for feature in RAW_DATA}
    return first_set, second_set

def tuples(data):
    ''' Helper function.
        data should be on the raw_data format '''
    features = [f for f in data]
    zipped = list(zip(*[data[f] for f in data]))
    return [{features[i]:z[i] for i in range(len(features))} for z in zipped]
    
def ratio(data, posterior_lambda, prior_lambda = None):
    ''' Calculates relative frequency of posterior vs prior.
        data should be on the raw_data format '''
    t = tuples(data)
    prior = t if not prior_lambda else [s for s in t if prior_lambda(s)]
    posterior = [s for s in prior if posterior_lambda(s)]
    return 0 if len(prior)==0 else len(posterior)/len(prior)

