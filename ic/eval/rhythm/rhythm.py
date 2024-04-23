# NOTE: copied from https://github.com/GabSpira/Rhythmic-Complexity-Metrics.git
import numpy as np
import math
from .util import prime_factors, lcm, get_pattern, next_power_of_2, check_window, \
                             get_IOI_frequencies, get_runs, get_ditribution, get_composites, entropy

class Metrics:
    def __init__(self, length, onsets_indeces):
        self.length = length
        self.onsets_indeces = onsets_indeces

    def __str__(self):
        return f"{self.length}({self.onsets_indeces})"
    
    def getToussaintComplexity(self):
        
        print('\n\n### TOUSSAINT ###')

        # Build hierarchy
        levels = int(math.log2(self.length))+1
        weights = np.zeros(self.length)
        for level in range(levels) :
            step = pow(2,level)
            weights[0:16:step] += 1
        print('The pattern has length equal to ', self.length, ', so the relative hierarchy is: ', weights)

        # Obtain non-normalized complexity score (metricity, inversely proportional to actual complexity)
        onset_weights = weights[self.onsets_indeces]
        metricity = sum(onset_weights)
        print('The relative metricity (inversely proportional to the complexity) is: ', metricity)

        # Obtain complexity score n1 (Onset Normalization) 
        n = (self.onsets_indeces).shape[0]                      #n° of onsets in the pattern
        n_sorted_weights = np.argsort(weights)[::-1][:n]
        max_metricity = sum(weights[n_sorted_weights])
        complexity_Toussaint_OnsetNorm = max_metricity - metricity
        print('The onset normalized Toussaint complexity score is: ', complexity_Toussaint_OnsetNorm)

        # Obtain complexity score n2 (Pulse Normalization)
        scale = sum(weights)
        complexity_Toussaint_PulseNorm = metricity / scale
        print('The pulse normalized Toussaint complexity score is ', complexity_Toussaint_PulseNorm)

        # Obtain complexity score n3 (Pulse-Onset Normalization)
        complexity_Toussaint_PulseOnsetNorm = complexity_Toussaint_PulseNorm/n
        print('The pulse-onset normalized Toussaint complexity score is ', complexity_Toussaint_PulseOnsetNorm)

        return(complexity_Toussaint_OnsetNorm, complexity_Toussaint_PulseNorm, complexity_Toussaint_PulseOnsetNorm)
    
    def getPalmerKrumhanslMUSComplexity(self):
        
        print('\n\n### PALMER & KRUMHANSL MUS ###')

        # Build hierarchy (intended by default only for 16-length patterns)
        weights_mus = np.array([6.1 , 4 , 4.25, 3.4, 5.9, 3, 4.4, 3.5, 5.75, 2.7, 3.1, 3.25, 5.4, 3.35, 4.9, 4.45])
        
        # Obtain metricity according to musician based weights
        onset_weights_mus = weights_mus[self.onsets_indeces]
        metricity_mus = sum(onset_weights_mus)
        print('According to the musician weights the metricity is: ', metricity_mus)

        # Obtain complexity score n1 (Onset Normalization)
        n = (self.onsets_indeces).shape[0]                      #n° of onsets in the pattern
        n_sorted_weights_mus = np.argsort(weights_mus)[::-1][:n]
        max_metricity_mus = sum(weights_mus[n_sorted_weights_mus])
        complexity_PalmerKrumhanslMUS_OnsetNorm = max_metricity_mus - metricity_mus
        print('The onset normalized Palmer&Krumhansl complexity score according to the musician-based weights is: ', complexity_PalmerKrumhanslMUS_OnsetNorm)

        # Obtain complexity score n2 (Pulse Normalization)
        scale = sum(weights_mus)
        complexity_PalmerKrumhanslMUS_PulseNorm = metricity_mus / scale
        print('The pulse normalized Palmer&Krumhansl complexity score according to the musician-based weights is ', complexity_PalmerKrumhanslMUS_PulseNorm)

        # Obtain complexity score n3 (Pulse-Onset Normalization)
        complexity_PalmerKrumhanslMUS_PulseOnsetNorm = complexity_PalmerKrumhanslMUS_PulseNorm/n
        print('The pulse-onset normalized Palmer&Krumhansl complexity score according to the musician-based weights is ', complexity_PalmerKrumhanslMUS_PulseOnsetNorm)

        return(complexity_PalmerKrumhanslMUS_OnsetNorm, complexity_PalmerKrumhanslMUS_PulseNorm, complexity_PalmerKrumhanslMUS_PulseOnsetNorm)
    
    def getPalmerKrumhanslNONMUSComplexity(self):
        
        print('\n\n### PALMER & KRUMHANSL NON MUS ###')

        # Build hierarchy (intended by default only for 16-length patterns)
        weights_nonmus = np.array([4.15, 3.45, 3.75, 3.75, 5.25, 3.4, 3.25, 3.8, 4.75, 4.5, 3.4, 4, 5.3, 3.9, 4.3, 4.45])
        
        # Obtain metricity according to non-musician based weights
        onset_weights_nonmus = weights_nonmus[self.onsets_indeces]
        metricity_nonmus = sum(onset_weights_nonmus)
        print('According to the non-musician weights the metricity is: ', metricity_nonmus)

        # Obtain complexity score n1 (Onset Normalization)
        n = (self.onsets_indeces).shape[0]                      #n° of onsets in the pattern
        n_sorted_weights_nonmus = np.argsort(weights_nonmus)[::-1][:n]
        max_metricity_nonmus = sum(weights_nonmus[n_sorted_weights_nonmus])
        complexity_PalmerKrumhanslNONMUS_OnsetNorm = max_metricity_nonmus - metricity_nonmus
        print('The onset normalized Palmer&Krumhansl complexity score according to the non-musician based weights is: ', complexity_PalmerKrumhanslNONMUS_OnsetNorm)

        # Obtain complexity score n2 (Pulse Normalization)
        scale = sum(weights_nonmus)
        complexity_PalmerKrumhanslNONMUS_PulseNorm = metricity_nonmus / scale
        print('The pulse normalized Palmer&Krumhansl complexity score according to the non-musician based weights is ', complexity_PalmerKrumhanslNONMUS_PulseNorm)

        # Obtain complexity score n3 (Pulse-Onset Normalization)
        complexity_PalmerKrumhanslNONMUS_PulseOnsetNorm = complexity_PalmerKrumhanslNONMUS_PulseNorm/n
        print('The pulse-onset normalized Palmer&Krumhansl complexity score according to the non-musician based weights is ', complexity_PalmerKrumhanslNONMUS_PulseOnsetNorm)

        return(complexity_PalmerKrumhanslNONMUS_OnsetNorm, complexity_PalmerKrumhanslNONMUS_PulseNorm, complexity_PalmerKrumhanslNONMUS_PulseOnsetNorm)
        
    def getEulerComplexity(self):
    
        print('\n\n### EULER ###')

        # Build hierarchy
        weights = np.zeros(self.length)
        for i in range(self.length) :
            m = lcm(i+self.length, self.length)/math.gcd(i+self.length, self.length)
            factors, pow = prime_factors(m)
            product = np.zeros(len(pow)).astype(int)
            for n in range(len(factors)):
                product[n] = (factors[n]-1)*pow[n]
            weights[i] = 1 + sum(product)
        print('The pattern has length equal to ', self.length, ', so the relative hierarchy is: ', weights)

        # Obtain complexity score (as written in Thul's Thesis, being inversely proportional to the actual complexity)
        onset_weights = weights[self.onsets_indeces]
        metricity = sum(onset_weights)
        print('The relative metricity (inversely proportional to the complexity) is: ', metricity)

        # Obtain complexity score (Onset Normalization)
        n = (self.onsets_indeces).shape[0]                      #n° of onsets in the pattern
        n_sorted_weights = np.argsort(weights)[::-1][:n]
        max_metricity = sum(weights[n_sorted_weights])
        complexity_Euler_OnsetNorm = max_metricity - metricity
        print('The Euler complexity score is: ', complexity_Euler_OnsetNorm)

        # Obtain complexity score n2 (Pulse Normalization)
        scale = sum(weights)
        complexity_Euler_PulseNorm = metricity / scale
        print('The pulse normalized Euler complexity score is ', complexity_Euler_PulseNorm)

        # Obtain complexity score n3 (Pulse-Onset Normalization)
        complexity_Euler_PulseOnsetNorm = complexity_Euler_PulseNorm/n
        print('The pulse-onset normalized Euler complexity score is ', complexity_Euler_PulseOnsetNorm)

        return(complexity_Euler_OnsetNorm, complexity_Euler_PulseNorm, complexity_Euler_PulseOnsetNorm)
    

    def getLonguetHigginsLeeComplexity(self):
        
        print('\n\n### LONGUET-HIGGINS & LEE ###')

        # Build hierarchy
        levels = int(math.log2(self.length))+1
        weights = np.zeros(self.length)
        for level in range(levels) :
            step = pow(2,level)
            for i in range(1,self.length):
                if i%step != 0:
                    weights[i] -= 1
        print('The pattern has length equal to ', self.length, ', so the relative hierarchy is: ', weights)
        
        # Build pattern
        pattern = get_pattern(self.length, self.onsets_indeces)

        # Find syncopations
        syncopations = []
        check = -10
        for i in range(self.length):
            if pattern[i] == 0:              #for all silences
                if weights[i]>check:            #if they have greater weight then the previous one (in case there has been syncopation)
                    search_zone = list(range(i-1, -1, -1)) + list(range(self.length-1, i, -1))
                    for j in search_zone:
                        if ((pattern[j] != 0) & (weights[i]>weights[j])):   #if there's an onset with lower weight
                            s = weights[i]-weights[j]
                            if s>0:                          #and it is situated before of the silence
                                syncopations.append(s)              #then there has been a syncopation
                            break
                check = weights[i]
        syncopations = np.array(syncopations)

        # Complexity score
        complexity_LonguetHiggindLee = sum(syncopations)
        print('The Longuet-Higgins & Lee complexity score is: ', complexity_LonguetHiggindLee)

        return(complexity_LonguetHiggindLee)
    
    def getFitchRosenfeldComplexity(self):
            
        print('\n\n### FITCH & ROSENFELD ###')

        # Build hierarchy
        levels = int(math.log2(self.length))+1
        weights = np.zeros(self.length)
        for level in range(levels) :
            step = pow(2,level)
            for i in range(1,self.length):
                if i%step != 0:
                    weights[i] -= 1
        print('The pattern has length equal to ', self.length, ', so the relative hierarchy is: ', weights)
        
        # Build pattern
        pattern = get_pattern(self.length, self.onsets_indeces)

        # Find syncopations
        syncopations = []                   #this method doesn't include a relative check
        for i in range(self.length):
            if pattern[i] == 0:             #for all silences
                search_zone = list(range(i-1, -1, -1)) + list(range(self.length-1, i, -1))
                for j in search_zone:
                    if ((pattern[j] != 0) & (weights[i]>weights[j])):   #if there's an onset with lower weight
                        s = weights[i]-weights[j]
                        if s>0:                          #and it is situated before of the silence
                            syncopations.append(s)              #then there has been a syncopation
                        break
        syncopations = np.array(syncopations)

        # Complexity score
        complexity_FitchRosenfeld = sum(syncopations)
        print('The Fitch & Rosenfeld complexity score is: ', complexity_FitchRosenfeld)

        return(complexity_FitchRosenfeld)
    
    def getSmithHoningComplexity(self):
            
        print('\n\n### SMITH & HONING ###')

        # Build hierarchy
        levels = int(math.log2(self.length))+1
        weights = np.zeros(self.length)
        for level in range(levels) :
            step = pow(2,level)
            for i in range(1,self.length):
                if i%step != 0:
                    weights[i] -= 1
        print('The pattern has length equal to ', self.length, ', so the relative hierarchy is: ', weights)
        
        # Build pattern
        pattern = get_pattern(self.length, self.onsets_indeces)

        # Find syncopations
        syncopations = []
        check = -10
        for i in range(self.length):
            if pattern[i] == 0:                          #for all silences
                if weights[i]>check:                     #if they have greater weight then the previous one in case there has been syncopation
                    search_zone = list(range(i-1, -1, -1)) + list(range(self.length-1, i, -1))
                    for j in search_zone:
                        if ((pattern[j] != 0) & (weights[i]>weights[j])):   #if there's an onset with lower weight
                            s = weights[i]-weights[j]+1
                            if s>1:                                         #and it is situated before of the silence
                                syncopations.append(s)                      #then there has been a syncopation
                            break
                check = weights[i]
        syncopations = np.array(syncopations)

        # Complexity score
        complexity_SmithHoning = sum(syncopations)
        print('The Smith & Honing complexity score is: ', complexity_SmithHoning)

        return(complexity_SmithHoning)
    
    def getPressingComplexity(self):

        print('\n\n### PRESSING ###')

        # Pattern initialization 
        pattern = get_pattern(self.length, self.onsets_indeces)
        
        # Get chunks of the pattern - by default intended for binary patterns
        chunk_dimensions = np.zeros(math.ceil(math.log2(self.length))).astype(int)
        metrical_levels = len(chunk_dimensions)
        for i in range(metrical_levels):
            chunk_dimensions[i] = int(self.length/math.pow(2, i))
            i +=1
        
        # Get the complexity as the sum of the averages of the chunk weights obtained in each metrical level
        avg = np.zeros(metrical_levels)
        for i in range(metrical_levels): 
            chunks = np.reshape(pattern, (-1, chunk_dimensions[i]))
            m,n = chunks.shape                          # The pattern is divided in m slices (sub-rhythms) of length n
            weights = np.zeros(m).astype(int)            
            for j in range(m):                          # for each sub-rhythm find the associate weight
                sub_rhythm = chunks[j,:]
                sub_sub_rhythm_1, sub_sub_rhythm_2 = np.split(sub_rhythm, 2)
                # print('metrical level: ', i, ', chunk number ', j, ': ', sub_sub_rhythm_1, sub_sub_rhythm_2)
                
                if ((sub_sub_rhythm_1[0]!=0) & (sub_sub_rhythm_2[0]!=0)):
                    weights[j] = 1                                              #filled
                if ((sub_rhythm[0]!=0) & (sub_rhythm[1]!=0)):
                    weights[j] = 2                                              #run
                if j+1==m:
                    if ((sub_rhythm[-1]!=0) & (chunks[0,0]!=0)):
                        weights[j] = 3                                          #upbeat
                elif j+1<m:
                    if ((sub_rhythm[-1]!=0) & (chunks[j+1,0]!=0)):
                        weights[j] = 3                                          #upbeat
                if ((sub_sub_rhythm_1[0]==0) or (sub_sub_rhythm_2[0]==0)):
                    weights[j] = 4                                              #syncopated
                if (1 not in sub_rhythm[1:-1]):                 
                    weights[j] = 0                                              #null
            
            avg[i] = np.sum(weights)/m
        complexity_Pressing = np.sum(avg)   

        print('The Pressing complexity score is: ', complexity_Pressing)

        return(complexity_Pressing)
    

    def getTanguianeComplexity(self):

        print('\n\n### TANGUIANE ###')

        # Pattern initialization 
        pattern = get_pattern(self.length, self.onsets_indeces)
        
        # Get chunks of the pattern - by default intended for binary patterns
        chunk_dimensions = np.zeros(math.ceil(math.log2(self.length))).astype(int)
        metrical_levels = len(chunk_dimensions)
        for i in range(metrical_levels):
            chunk_dimensions[i] = int(self.length/math.pow(2, i))
            i +=1
        
        # Get the complexity as the max number of root patterns in a metrical level
        n_root_patterns = np.zeros(metrical_levels)
        for i in range(metrical_levels): 
            chunks = np.reshape(pattern, (-1, chunk_dimensions[i]))
            m,n = chunks.shape                          # The pattern is divided in m slices (sub-rhythms) of length n           
            roots_onset = []
            roots_silence = []
            for j in range(m):                          # for each sub-rhythm compare it to all the others
                test1 = chunks[j,:]
                if test1[0]!=0:                         # separating when they start with an onset
                    for k in range(m):
                        test2 = chunks[k,:]
                        if test2[0]!=0:
                            root_pattern = np.bitwise_and(test1, test2)
                            if (j!=k & ((root_pattern==test1).all() or (root_pattern==test2).all())):
                                roots_onset.append(root_pattern)     
                if test1[0]==0:                         # from when they start with a silence
                    test1 = np.bitwise_not(test1)
                    for k in range(m):
                        test2 = chunks[k,:]
                        if test2[0]==0:
                            test2 = np.bitwise_not(test2)
                            root_pattern = np.bitwise_and(test1, test2)
                            if (j!=k & ((root_pattern==test1).all() or (root_pattern==test2).all())):
                                roots_silence.append(root_pattern) 
                                
                roots_score_1 = len(np.unique(np.array(roots_silence)))  
                roots_score_2 = len(np.unique(np.array(roots_onset)))          
                
            n_root_patterns[i] = roots_score_1 + roots_score_2
            
        complexity_Tanguiane = np.max(n_root_patterns)   
        print('The Tanguiane complexity score is: ', complexity_Tanguiane)
        
        return(complexity_Tanguiane)
    

    def getKeithComplexity(self):

        print('\n\n### KEITH ###')

        complexity_Keith = 0                            #init

        # For each couple of onset indeces compute their distance to know if their pattern is hesitation, anticipation or syncopation
        for k in range(len(self.onsets_indeces)):
            i = self.onsets_indeces[k]
            if k==len(self.onsets_indeces)-1: j = self.onsets_indeces[0]+self.length
            elif k<len(self.onsets_indeces)-1: j = self.onsets_indeces[k+1]
            delta = j - i
            delta_hat = next_power_of_2(delta)
            i_displacement = i % delta_hat          # if equal to 0 it means i is on the beat otherwise it's off the beat
            j_displacement = j % delta_hat          # if equal to 0 it means j is on the beat otherwise it's off the beat
            if ((i_displacement == 0) & (j_displacement == 0)):     s = 0       #meter    
            if ((i_displacement == 0) & (j_displacement != 0)):     s = 1       #hesitation
            if ((i_displacement != 0) & (j_displacement == 0)):     s = 2       #anticipation
            if ((i_displacement != 0) & (j_displacement != 0)):     s = 3       #syncopation
            complexity_Keith = complexity_Keith + s

        print('The Keith complexity score is: ', complexity_Keith)
        return(complexity_Keith)
    

    def getDirectedSwapDistanceComplexity(self):

        print('\n\n### DIRECTED SWAP DISTANCE ###')

        # Meter initialization - intended by default for 16-length rhythms
        meter2_indeces = [0,2,4,6,8,10,12,14]
        meter4_indeces = [0,4,8,12]
        meter8_indeces = [0,8]

        # Distance initialization for each meter
        distance_meter2 = 0
        distance_meter4 = 0
        distance_meter8 = 0

        # Compute directed swap distance for each reference meter
        for i in range(len(self.onsets_indeces)):
            distance_meter2 = distance_meter2 + min(abs(meter2_indeces - self.onsets_indeces[i]))
            distance_meter4 = distance_meter4 + min(abs(meter4_indeces - self.onsets_indeces[i]))
            distance_meter8 = distance_meter8 + min(abs(meter8_indeces - self.onsets_indeces[i]))

        complexity_DirectedSwapDistance_m2 = distance_meter2
        complexity_DirectedSwapDistance_m4 = distance_meter4
        complexity_DirectedSwapDistance_m8 = distance_meter8   
        complexity_DirectedSwapDistance_mean = (distance_meter8 + distance_meter4 + distance_meter2)/3

        print('The Directed Swap Distance complexity score referred to a meter with 8 pulses is: ', complexity_DirectedSwapDistance_m2)
        print('The Directed Swap Distance complexity score referred to a meter with 4 pulses is: ', complexity_DirectedSwapDistance_m4)
        print('The Directed Swap Distance complexity score referred to a meter with 2 pulses is: ', complexity_DirectedSwapDistance_m8)
        print('The mean Directed Swap Distance complexity score is: ', complexity_DirectedSwapDistance_mean)

        return(complexity_DirectedSwapDistance_m2, complexity_DirectedSwapDistance_m4,
               complexity_DirectedSwapDistance_m8, complexity_DirectedSwapDistance_mean) 
    
    def getWeightedNotetoBeatDistance(self):
        
        print('\n\n### WEIGHTED NOTE TO BEAT DISTANCE ###')

        # Meter initialization - intended by default for 16-length rhythms
        meter4_indeces_2bars = [0,4,8,12,16,20,24,28]
        sum_weights = 0

        # For each onset compute the weights depending on the distance from the nearest beat and the following ones
        for i in range(len(self.onsets_indeces)):
            
            # define the considered onset
            x = self.onsets_indeces[i]                          # index of the considered onset in the pattern array
            
            # define the smaller distance from a beat and its index
            d = np.min(abs(meter4_indeces_2bars - x))              # n° of onsets btw the considered onset and the nearest beat
            T = d/len(meter4_indeces_2bars)                        # actual distance

            # define where the considered onset ends
            if i+1<len(self.onsets_indeces): 
                end = self.onsets_indeces[i+1]
            else: 
                end = self.length + self.onsets_indeces[0]
                
            # define the beats after the considered onset
            for k in range(len(meter4_indeces_2bars)):
                if meter4_indeces_2bars[k]>x: 
                    e1 = meter4_indeces_2bars[k]                   # first beat after the considered onset
                    e2 = meter4_indeces_2bars[k+1]                 # second beat after the considered onset
                    break            

            # assign weights based on the previous parameters
            if ((end <= e1) & (T!=0)):
                D = 1/T
            elif ((end <= e2)  & (T!=0)):
                D = 2/T
            elif ((e2 < end)  & (T!=0)):
                D = 1/T
            elif T==0: 
                D=0

            sum_weights = sum_weights + D

        # Complexity score
        complexity_WNBD = sum_weights/len(self.onsets_indeces)     
        print('The Weighted Note to Beat Distance complexity score is: ', complexity_WNBD)

        return(complexity_WNBD) 

        

    def getHkSpanComplexity(self):

        print('\n\n### H (k-span) ###')

        # simple version (only log2 of the sequence)
        #TODO: UPDATE THIS FOR PATTERNS WITH DIFFERENT VELOCITIES

        complexity_HkSpan = math.log2(self.length)
        print('The H (k-span) complexity score is: ', complexity_HkSpan)

        return(complexity_HkSpan)
    
    def getHrunSpanComplexity(self):
        
        print('\n\n### H (run-span) ###')

        # simple version (only log2 of the sequence)
        #TODO: UPDATE THIS FOR PATTERNS WITH DIFFERENT VELOCITIES

        # pattern initialization
        pattern = get_pattern(self.length, self.onsets_indeces)

        current_run = []
        runs = []

        # build runs
        for i in range(len(pattern)):
            # check run end
            if i == 0 or pattern[i]!=pattern[i-1]:
                if len(current_run)>=2:
                    runs.append(current_run)
                current_run = []
            # continue run
            current_run.append([pattern[i]])
        # add last run 
        if len(current_run)>=2: 
            runs.append(current_run)

        complexity_HrunSpan = math.log2(len(runs))
        print('The H (run-span) complexity score is: ', complexity_HrunSpan)

        return(complexity_HrunSpan)
    

    def getCEPSComplexity(self):

        print('\n\n### CODED ELEMENT PROCESSING SYSTEM###')
        #TODO: conclude with joint entropy + update for velocity patterns

        # pattern = get_pattern(self.length, self.onsets_indeces)
        pattern = get_pattern(8, [1,2,4,5,7])

        # Starting representation (level 1)

        # Value variable (x)
        level = pattern
        level_distribution = np.bincount(level) / len(level) 
        # Length variable    
        lengths_in_level = [len(str(level[i])) for i in range(len(level))]
        lengths_distribution = np.bincount(lengths_in_level) /len(lengths_in_level)

        level_order = 1
        complexity_CEPS = 0

        while (True):

            # (Uncomment below to print level informations)

            # print('\n## LEVEL ', level_order)
            # print('value var: ', level, ' with distribution: ', level_distribution)
            # print('len var: ', lengths_in_level, ' with distribution: ', lengths_distribution)
        
            #entropies
            H_max_level = entropy(level_distribution, lengths_distribution)
            # print('entropy of level: ', H_max_level)
            complexity_CEPS += H_max_level

            level_order += 1

            if(len(level)==1): break                    # end if the whole pattern has been constructed as a single composite
            
            # Value variable (x)
            if (level_order % 2 == 0):                  # even levels are obtained by finding runs
                level = get_runs(level)
            else: 
                level = get_composites(level)           # odd levels are obtained by finding composites
            level_distribution = get_ditribution(level)

            # Length variable (y)
            lengths_in_level = [len((level[i])) for i in range(len(level))]
            lengths_distribution = np.bincount(lengths_in_level) / len(lengths_in_level)
            
        print('The CEPS complexity score is: ', complexity_CEPS)

        return complexity_CEPS
    

    
    def getLempelZivCodingComplexity(self):

        print('\n\n### LEMPEL-ZIV CODING ###')

        # Variables initialization
        s = []                                              # vocabulary
        r = get_pattern(self.length, self.onsets_indeces)   # pattern
        r = np.concatenate((r,r))                           # 2 iterations
        j = 0                                               # window start
        i = 0                                               # window end
        
        # Build the vocabulary
        while (True):
            q = r[j:i+1]                                    #window  
            can_generate_q_from_s = check_window(q, s)
            if (can_generate_q_from_s == False): 
                s.append(q)
                i += 1
                j = i
            else: i += 1
            if i == (self.length * 2): break
        print('The pattern can be constructed from this vocabulary of sub-patterns: ', s)

        # Derive the complexity as the length of the vocabulary
        complexity_LempelZiv = len(s)     
        print('The Lempel Ziv complexity score is: ', complexity_LempelZiv)

        return complexity_LempelZiv

            
    def getStandardDeviationComplexity(self):

        print('\n\n### IOI - STANDARD DEVIATION ###')

        # Get IOI frequencies, both global and local
        global_frequencies, local_frequencies = get_IOI_frequencies(self.length, self.onsets_indeces)

        # Compute Standard Deviation - global IOIs
        std_global = np.std(global_frequencies)

        # Compute Standard Deviation - local IOIs
        std_local = np.std(local_frequencies)

        # Compute complexities as inversely proportional to STDs
        complexity_STD_globalIOI = 1/std_global
        complexity_STD_localIOI = 1/std_local
        print('The global IOIs Standard Deviation complexity score is: ', complexity_STD_globalIOI)
        print('The local IOIs Standard Deviation complexity score is: ', complexity_STD_localIOI)
        
        return complexity_STD_globalIOI, complexity_STD_localIOI
    
    def getInformationEntropyComplexity(self):

        print('\n\n### IOI - INFORMATION ENTROPY ###')

        # Get IOI frequencies, both global and local
        global_frequencies, local_frequencies = get_IOI_frequencies(self.length, self.onsets_indeces)
        
        # Compute probability distributions
        global_pdf = global_frequencies/np.sum(global_frequencies)
        local_pdf = local_frequencies/np.sum(local_frequencies)

        # Compute entropies
        H_global = 0
        for i in range(len(global_pdf)):
            if global_pdf[i] != 0:
                H_global -= global_pdf[i]*math.log2(global_pdf[i])
        H_local = 0
        for i in range(len(local_pdf)):
            if local_pdf[i] != 0:
                H_local -= local_pdf[i]*math.log2(local_pdf[i])

        # Complexities
        complexity_InformationEntropy_globalIOI = H_global
        complexity_InformationEntropy_localIOI = H_local
        print('The global IOIs Information Entropy complexity score is: ', complexity_InformationEntropy_globalIOI)
        print('The local IOIs Information Entropy complexity score is: ', complexity_InformationEntropy_localIOI)
        
        return complexity_InformationEntropy_globalIOI, complexity_InformationEntropy_localIOI
    
    
    def getTallestBinComplexity(self):

        print('\n\n### IOI - TALLEST BIN ###')

        # Get IOI frequencies, both global and local
        global_frequencies, local_frequencies = get_IOI_frequencies(self.length, self.onsets_indeces)

        # Compute probability distributions
        global_pdf = global_frequencies/np.sum(global_frequencies)
        local_pdf = local_frequencies/np.sum(local_frequencies)     

        # Find the max in each distribution
        global_max = np.max(global_pdf)
        local_max = np.max(local_pdf)

        # Compute the complexities 
        complexity_TallestBin_globalIOI = 1/global_max
        complexity_TallestBin_localIOI = 1/local_max

        print('The global IOIs Tallest Bin complexity score is: ', complexity_TallestBin_globalIOI)
        print('The local IOIs Tallest Bin complexity score is: ', complexity_TallestBin_localIOI)

        return(complexity_TallestBin_globalIOI, complexity_TallestBin_localIOI)


    def getOffBeatnessComplexity(self):

        print('\n\n### TOUSSAINT OFF-BEATNESS ###')
        
        # Find the possibly inscribible polygons
        polygon_vertices = []
        for i in range(2,self.length):
            if self.length%i==0: polygon_vertices.append(i)

        # Draw the polygons (mark the on-beat pulses)
        on_beat_indeces = []
        for i in polygon_vertices:
            for j in range(self.length):
                if ((j*i<self.length) & (j*i not in on_beat_indeces)):
                    on_beat_indeces.append(j*i)
        
        # Derive the off-beat pulses
        off_beat_indeces = np.setdiff1d(np.arange(self.length), on_beat_indeces)
        
        # Find the complexity as the number of onsets that are off-beat
        complexity_OffBeatness = 0
        for i in self.onsets_indeces:
            if i in off_beat_indeces: complexity_OffBeatness += 1

        print('The Off-Beatness complexity score is: ', complexity_OffBeatness)

        return(complexity_OffBeatness)
    

    def getRhythmicOddityComplexity(self):

        print('\n\n### RHYTHMIC ODDITY ###')

        # Find the pairs of onsets that divide in 2 the pattern
        total = 0
        for i in range(int(self.length/2)):
            if ((i in self.onsets_indeces) & (int(i+self.length/2) in self.onsets_indeces)): 
                total += 1

        # Derive the complexity
        if total != 0:
            complexity_RhythmicOddity = 1/total
        else:
            complexity_RhythmicOddity = 0

        print('The Rhythmic Oddity complexity score is: ', complexity_RhythmicOddity)

        return(complexity_RhythmicOddity)
