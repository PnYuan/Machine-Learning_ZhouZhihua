# -*- coding: utf-8 -*
    
'''''
create on 2017/3/24, the day after our national football team beat south korea

@author: PY131
'''''

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
class Node(object): 
    def __init__(self, attr_init=None, label_init=None, attr_down_init={} ):  
        self.attr = attr_init  
        self.label = label_init 
        self.attr_down = attr_down_init 
    
''' 
Branching for decision tree using recursion 
 
@param df: the pandas dataframe of the data_set
@return root: Node, the root node of decision tree
'''  
def TreeGenerate(df):
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
        new_node.attr, div_value = OptAttr(df)
        
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
    
'''
make a predict based on root

@param root: Node, root Node of the decision tree
@param df_sample: dataframe, a sample line 
'''
def Predict(root, df_sample):   
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

'''
calculating the appeared label and it's counts
 
@param label_arr: data array for class labels
@return label_count: dict, the appeared label and it's counts
'''  
def NodeLabel(label_arr):
    label_count = {}       # store count of label 
      
    for label in label_arr:
        if label in label_count: label_count[label] += 1
        else: label_count[label] = 1
        
    return label_count

'''
calculating the appeared value for categoric attribute and it's counts

@param data_arr: data array for an attribute
@return value_count: dict, the appeared value and it's counts
'''
def ValueCount(data_arr):
    value_count = {}       # store count of value 
      
    for label in data_arr:
        if label in value_count: value_count[label] += 1
        else: value_count[label] = 1
        
    return value_count

'''
find the optimal attributes of current data_set
 
@param df: the pandas dataframe of the data_set 
@return opt_attr:  the optimal attribution for branch
@return div_value: for discrete variable value = 0
                   for continuous variable value = t for bisection divide value
'''  
def OptAttr(df):
    info_gain = 0
    
    for attr_id in df.columns[1:-1]:
        info_gian_tmp, div_value_tmp = InfoGain(df, attr_id)
        if  info_gian_tmp > info_gain : 
            info_gain = info_gian_tmp
            opt_attr = attr_id
            div_value = div_value_tmp
        
    return opt_attr, div_value
        
'''
calculating the information gain of an attribution
 
@param df:      dataframe, the pandas dataframe of the data_set
@param attr_id: the target attribution in df
@return info_gain: the information gain of current attribution
@return div_value: for discrete variable, value = 0
               for continuous variable, value = t (the division value)
'''  
def InfoGain(df, index):
    info_gain = InfoEnt(df.values[:,-1])  # info_gain for the whole label
    div_value = 0  # div_value for continuous attribute
    
    n = len(df[index])  # the number of sample
    # 1.for continuous variable using method of bisection
    if df[index].dtype == (float, int):
        sub_info_ent = {}  # store the div_value (div) and it's subset entropy
        
        df = df.sort([index], ascending=1)  # sorting via column
        df = df.reset_index(drop=True)
        
        data_arr = df[index]
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
        data_arr = df[index]
        label_arr = df[df.columns[-1]]
        value_count = ValueCount(data_arr)
            
        for key in value_count:
            key_label_arr = label_arr[data_arr == key]
            info_gain -= value_count[key] * InfoEnt(key_label_arr) / n
    
    return info_gain, div_value
    
'''
calculating the information entropy of an attribution
 
@param label_arr: ndarray, class label array of data_arr
@return ent: the information entropy of current attribution
'''  
def InfoEnt(label_arr):
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

