# -*- coding: utf-8 -*
    
'''''
@author: PY131
'''''

class Node(object): 
    '''
    definition of decision node class
    
    attr: attribution as parent for a new branching 
    attr_down: dict: {key, value}
            key:   categoric:  categoric attr_value 
                   continuous: '<= div_value' for small part
                               '> div_value' for big part
            value: children (Node class)
    label： class label (the majority of current sample labels)
    ''' 
    def __init__(self, attr_init=None, label_init=None, attr_down_init={} ):  
        self.attr = attr_init  
        self.label = label_init 
        self.attr_down = attr_down_init 
    
def TreeGenerate(df):
    ''' 
    Branching for decision tree using recursion 
     
    @param df: the pandas dataframe of the data_set
    @return root: Node, the root node of decision tree
    '''  
    # generating a new root node
    new_node = Node(None, None, {})
    label_arr = df[df.columns[-1]]
    
    label_count = NodeLabel(label_arr)
    if label_count:  # assert the label_count isn's empty
        new_node.label= max(label_count, key=label_count.get) 
            
        # end if there is only 1 class in current node data
        # end if attribution array is empty
        if len(label_count) == 1 or len(label_arr) == 0:
            return new_node
        
        # get the optimal attribution for a new branching
        new_node.attr, div_value = OptAttr_Gini(df)  # via Gini index 
        
        # recursion
        if div_value == 0:  # categoric variable
            value_count = ValueCount(df[new_node.attr]) 
            for value in value_count:
                df_v = df[ df[new_node.attr].isin([value]) ]  # get sub set
                # delete current attribution
                df_v = df_v.drop(new_node.attr, 1)  
                new_node.attr_down[value] = TreeGenerate(df_v)
                
        else:  # continuous variable # left and right child
            value_l = "<=%.3f" % div_value
            value_r = ">%.3f" % div_value
            df_v_l = df[ df[new_node.attr] <= div_value ]  # get sub set
            df_v_r = df[ df[new_node.attr] > div_value ]
 
            new_node.attr_down[value_l] = TreeGenerate(df_v_l)
            new_node.attr_down[value_r] = TreeGenerate(df_v_r)
        
    return new_node
    
def Predict(root, df_sample):   
    '''
    make a predict based on root
    
    @param root: Node, root Node of the decision tree
    @param df_sample: dataframe, a sample line 
    '''
    try :
        import re # using Regular Expression to get the number in string
    except ImportError :
        print("module re not found")
    
    while root.attr != None :        
        # continuous variable
        if df_sample[root.attr].dtype == (float, int):
            # get the div_value from root.attr_down
            for key in list(root.attr_down):
                num = re.findall(r"\d+\.?\d*",key)
                div_value = float(num[0])
                break
            if df_sample[root.attr].values[0] <= div_value:
                key = "<=%.3f" % div_value
                root = root.attr_down[key]
            else:
                key = ">%.3f" % div_value
                root = root.attr_down[key]
                
        # categoric variable
        else:  
            key = df_sample[root.attr].values[0]
            # check whether the attr_value in the child branch
            if key in root.attr_down: 
                root = root.attr_down[key]
            else: 
                break
            
    return root.label 

def PredictAccuracy(root, df_test):  
    '''
    calculating accuracy of prediction on test set
    
    @param root: Node, root Node of the decision tree
    @param df_test: dataframe, test data set
    @return accuracy, float,
    '''
    if len(df_test.index) == 0: return 0
    pred_true = 0
    for i in df_test.index:
        label = Predict(root, df_test[df_test.index == i])
        if label == df_test[df_test.columns[-1]][i]:
            pred_true += 1
    return pred_true / len(df_test.index)
    
def PrePurn(df_train, df_test):
    '''
    pre-purning to generating a decision tree
    
    @param df_train: dataframe, the training set to generating a tree
    @param df_test: dataframe, the testing set for purning decision
    @return root: Node, root of the tree using purning
    '''
    # generating a new root node
    new_node = Node(None, None, {})
    label_arr = df_train[df_train.columns[-1]]
    
    label_count = NodeLabel(label_arr)
    if label_count:  # assert the label_count isn's empty
        new_node.label= max(label_count, key=label_count.get) 
     
        # end if there is only 1 class in current node data
        # end if attribution array is empty
        if len(label_count) == 1 or len(label_arr) == 0:
            return new_node
        
        # calculating the test accuracy up to current node
        a0 = PredictAccuracy(new_node, df_test)
        
        # get the optimal attribution for a new branching
        new_node.attr, div_value = OptAttr_Gini(df_train)  # via Gini index 
        
        # get the new branch
        if div_value == 0:  # categoric variable
            value_count = ValueCount(df_train[new_node.attr]) 
            for value in value_count: 
                df_v = df_train[ df_train[new_node.attr].isin([value]) ]  # get sub set
                df_v = df_v.drop(new_node.attr, 1)
                # for child node
                new_node_child = Node(None, None, {}) 
                label_arr_child = df_train[df_v.columns[-1]]
                label_count_child = NodeLabel(label_arr_child)
                new_node_child.label = max(label_count_child, key=label_count_child.get)
                new_node.attr_down[value] = new_node_child
            
            # calculating to check whether need further branching
            a1 = PredictAccuracy(new_node, df_test)
            if a1 > a0:  # need branching
                for value in value_count: 
                    df_v = df_train[ df_train[new_node.attr].isin([value]) ]  # get sub set
                    df_v = df_v.drop(new_node.attr, 1)
                    new_node.attr_down[value] = TreeGenerate(df_v)
            else:
                new_node.attr = None
                new_node.attr_down = {}
                
        else:  # continuous variable # left and right child
            value_l = "<=%.3f" % div_value
            value_r = ">%.3f" % div_value
            df_v_l = df_train[ df_train[new_node.attr] <= div_value ]  # get sub set
            df_v_r = df_train[ df_train[new_node.attr] > div_value ]
 
            # for child node
            new_node_l = Node(None, None, {})
            new_node_r = Node(None, None, {})
            label_count_l = NodeLabel(df_v_l[df_v_r.columns[-1]])
            label_count_r = NodeLabel(df_v_r[df_v_r.columns[-1]])
            new_node_l.label = max(label_count_l, key=label_count_l.get)
            new_node_r.label = max(label_count_r, key=label_count_r.get)
            new_node.attr_down[value_l] = new_node_l
            new_node.attr_down[value_r] = new_node_r
            
            # calculating to check whether need further branching
            a1 = PredictAccuracy(new_node, df_test)
            if a1 > a0:  # need branching
                new_node.attr_down[value_l] = TreeGenerate(df_v_l)
                new_node.attr_down[value_r] = TreeGenerate(df_v_r)
            else:
                new_node.attr = None
                new_node.attr_down = {}
    
    return new_node
            
def PostPurn(root, df_test):
    '''
    pre-purning to generating a decision tree
    
    @param root: Node, root of the tree
    @param df_test: dataframe, the testing set for purning decision
    @return accuracy score through traversal the tree
    '''
    # leaf node
    if root.attr == None:  
        return PredictAccuracy(root, df_test)
    
    # calculating the test accuracy on children node
    a1 = 0
    value_count = ValueCount(df_test[root.attr]) 
    for value in list(value_count):
        df_test_v = df_test[ df_test[root.attr].isin([value]) ]  # get sub set
        if value in root.attr_down:  # root has the value
            a1_v = PostPurn(root.attr_down[value], df_test_v)
        else:  # root doesn't have value
            a1_v = PredictAccuracy(root, df_test_v)
        if a1_v == -1:  # -1 means no pruning back from this child
            return -1
        else:
            a1 += a1_v * len(df_test_v.index) / len(df_test.index)
            
    # calculating the test accuracy on this node   
    node = Node(None, root.label, {})
    a0 = PredictAccuracy(node, df_test)
    
    # check if need pruning
    if a0 >= a1:
        root.attr = None
        root.attr_down = {}
        return a0
    else: 
        return -1
   
def NodeLabel(label_arr):
    '''
    calculating the appeared label and it's counts
     
    @param label_arr: data array for class labels
    @return label_count: dict, the appeared label and it's counts
    '''  
    label_count = {}       # store count of label 
      
    for label in label_arr:
        if label in label_count: label_count[label] += 1
        else: label_count[label] = 1
        
    return label_count

def ValueCount(data_arr):
    '''
    calculating the appeared value for categoric attribute and it's counts
    
    @param data_arr: data array for an attribute
    @return value_count: dict, the appeared value and it's counts
    '''
    value_count = {}       # store count of value 
      
    for label in data_arr:
        if label in value_count: value_count[label] += 1
        else: value_count[label] = 1
        
    return value_count

'''
optimal attribution selection in CART algorithm based on gini index
'''
def OptAttr_Gini(df):
    '''
    find the optimal attributes of current data_set based on gini index
     
    @param df: the pandas dataframe of the data_set 
    @return opt_attr:  the optimal attribution for branch
    @return div_value: for discrete variable value = 0
                       for continuous variable value = t for bisection divide value
    '''
    gini_index = float('Inf')
    for attr_id in df.columns[1:-1]:
        gini_index_tmp, div_value_tmp = InfoGain(df, attr_id)
        if  gini_index_tmp < gini_index : 
            gini_index = gini_index_tmp
            opt_attr = attr_id
            div_value = div_value_tmp
        
    return opt_attr, div_value

def GiniIndex(df, attr_id):
    '''
    calculating the gini index of an attribution
     
    @param df:      dataframe, the pandas dataframe of the data_set
    @param attr_id: the target attribution in df
    @return gini_index: the gini index of current attribution
    @return div_value: for discrete variable, value = 0
                   for continuous variable, value = t (the division value)
    '''  
    gini_index = 0  # info_gain for the whole label
    div_value = 0  # div_value for continuous attribute
    
    n = len(df[attr_id])  # the number of sample
    
    # 1.for continuous variable using method of bisection
    if df[attr_id].dtype == (float, int):
        sub_gini = {}  # store the div_value (div) and it's subset gini value
        
        df = df.sort([attr_id], ascending=1)  # sorting via column
        df = df.reset_index(drop=True)
        
        data_arr = df[attr_id]
        label_arr = df[df.columns[-1]]
        
        for i in range(n-1):
            div = (data_arr[i] + data_arr[i+1]) / 2
            sub_gini[div] = ( (i+1) * Gini(label_arr[0:i+1]) / n ) \
                              + ( (n-i-1) * Gini(label_arr[i+1:-1]) / n )
        # our goal is to get the min subset entropy sum and it's divide value
        div_value, gini_index = min(sub_gini.items(), key=lambda x: x[1])
        
    # 2.for discrete variable (categoric variable)
    else:
        data_arr = df[attr_id]
        label_arr = df[df.columns[-1]]
        value_count = ValueCount(data_arr)
            
        for key in value_count:
            key_label_arr = label_arr[data_arr == key]
            gini_index += value_count[key] * Gini(key_label_arr) / n
    
    return gini_index, div_value

def Gini(label_arr):
    '''
    calculating the gini value of an attribution
     
    @param label_arr: ndarray, class label array of data_arr
    @return gini: the information entropy of current attribution
    '''     
    gini = 1
    
    n = len(label_arr)
    label_count = NodeLabel(label_arr)
    for key in label_count:
        gini -= ( label_count[key] / n ) * ( label_count[key] / n )
    
    return gini

'''
optimal attribution selection in ID3 algorithm based on information entropy
'''
def OptAttr_Ent(df):
    '''
    find the optimal attributes of current data_set based on info entropy
     
    @param df: the pandas dataframe of the data_set 
    @return opt_attr:  the optimal attribution for branch
    @return div_value: for discrete variable value = 0
                       for continuous variable value = t for bisection divide value
    '''  
    info_gain = 0
    
    for attr_id in df.columns[1:-1]:
        info_gian_tmp, div_value_tmp = InfoGain(df, attr_id)
        if  info_gian_tmp > info_gain : 
            info_gain = info_gian_tmp
            opt_attr = attr_id
            div_value = div_value_tmp
        
    return opt_attr, div_value
        
def InfoGain(df, attr_id):
    '''
    calculating the information gain of an attribution
     
    @param df:      dataframe, the pandas dataframe of the data_set
    @param attr_id: the target attribution in df
    @return info_gain: the information gain of current attribution
    @return div_value: for discrete variable, value = 0
                   for continuous variable, value = t (the division value)
    '''  
    info_gain = InfoEnt(df.values[:,-1])  # info_gain for the whole label
    div_value = 0  # div_value for continuous attribute
    
    n = len(df[attr_id])  # the number of sample
    # 1.for continuous variable using method of bisection
    if df[attr_id].dtype == (float, int):
        sub_info_ent = {}  # store the div_value (div) and it's subset entropy
        
        df = df.sort([attr_id], ascending=1)  # sorting via column
        df = df.reset_index(drop=True)
        
        data_arr = df[attr_id]
        label_arr = df[df.columns[-1]]
        
        for i in range(n-1):
            div = (data_arr[i] + data_arr[i+1]) / 2
            sub_info_ent[div] = ( (i+1) * InfoEnt(label_arr[0:i+1]) / n ) \
                              + ( (n-i-1) * InfoEnt(label_arr[i+1:-1]) / n )
        # our goal is to get the min subset entropy sum and it's divide value
        div_value, sub_info_ent_max = min(sub_info_ent.items(), key=lambda x: x[1])
        info_gain -= sub_info_ent_max
        
    # 2.for discrete variable (categoric variable)
    else:
        data_arr = df[attr_id]
        label_arr = df[df.columns[-1]]
        value_count = ValueCount(data_arr)
            
        for key in value_count:
            key_label_arr = label_arr[data_arr == key]
            info_gain -= value_count[key] * InfoEnt(key_label_arr) / n
    
    return info_gain, div_value
     
def InfoEnt(label_arr):
    '''
    calculating the information entropy of an attribution
     
    @param label_arr: ndarray, class label array of data_arr
    @return ent: the information entropy of current attribution
    ''' 
    try :
        from math import log2
    except ImportError :
        print("module math.log2 not found")
    
    ent = 0
    n = len(label_arr)
    label_count = NodeLabel(label_arr)
    
    for key in label_count:
        ent -= ( label_count[key] / n ) * log2( label_count[key] / n )
    
    return ent

def DrawPNG(root, out_file):
    '''
    visualization of decision tree from root.
    @param root: Node, the root node for tree.
    @param out_file: str, name and path of output file
    '''
    try:
        from pydotplus import graphviz
    except ImportError:
        print("module pydotplus.graphviz not found")
        
    g = graphviz.Dot()  # generation of new dot   

    TreeToGraph(0, g, root)
    g2 = graphviz.graph_from_dot_data( g.to_string() )
    
    g2.write_png(out_file) 
  
def TreeToGraph(i, g, root):
    '''
    build a graph from root on
    @param i: node number in this tree
    @param g: pydotplus.graphviz.Dot() object
    @param root: the root node
    
    @return i: node number after modified  
#     @return g: pydotplus.graphviz.Dot() object after modified
    @return g_node: the current root node in graphviz
    '''
    try:
        from pydotplus import graphviz
    except ImportError:
        print("module pydotplus.graphviz not found")

    if root.attr == None:
        g_node_label = "Node:%d\n好瓜:%s" % (i, root.label)
    else:
        g_node_label = "Node:%d\n好瓜:%s\n属性:%s" % (i, root.label, root.attr)
    g_node = i
    g.add_node( graphviz.Node( g_node, label = g_node_label ) )
    
    for value in list(root.attr_down):
        i, g_child = TreeToGraph(i+1, g, root.attr_down[value])
        g.add_edge( graphviz.Edge(g_node, g_child, label = value) ) 

    return i, g_node

