# Data Format

X -- n-dimensional list (not np array!) of all vector members, where the index represents the ID
of a member and the within list represents a members feature vector.

Example:
X = [[1,3,[1,0,0], [2,4,[1,0,0]], where the element 1 and 2 are homogenous feautures, the elements 3 and 4 are heterogenous features and the list [1,0,0] represent a categorical feature.

Here [1,3,[1,0,0]] is the feature vecotr of member 0 and [[2,4,[1,0,0]] is the feature vecotr of member 1.

homogenous -- 2-dimensional list (can be an np array), where the firs index is the ID of a member and the within list represents the members homohenous feature vector.
Example: [[1],[2]]

heterogenous -- 2-dimensional list (can be an np array), where the firs index is the ID of a member and the within list represents the members heterogenous feature vector.
Example: [[3],[4]]

one_hot -- 3-dimensional list, where the first index represents the ID of a member, the second index is the first one_hot feature of this member and the third index is the corresponding value.