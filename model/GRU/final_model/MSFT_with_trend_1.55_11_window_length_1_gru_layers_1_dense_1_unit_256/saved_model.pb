??*
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.6.02unknown8??(
|
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_14/kernel
u
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel* 
_output_shapes
:
??*
dtype0
s
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_14/bias
l
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes	
:?*
dtype0
{
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_15/kernel
t
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes
:	?*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
gru_14/gru_cell_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namegru_14/gru_cell_14/kernel
?
-gru_14/gru_cell_14/kernel/Read/ReadVariableOpReadVariableOpgru_14/gru_cell_14/kernel*
_output_shapes
:	?*
dtype0
?
#gru_14/gru_cell_14/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#gru_14/gru_cell_14/recurrent_kernel
?
7gru_14/gru_cell_14/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_14/gru_cell_14/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
gru_14/gru_cell_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_namegru_14/gru_cell_14/bias
?
+gru_14/gru_cell_14/bias/Read/ReadVariableOpReadVariableOpgru_14/gru_cell_14/bias*
_output_shapes
:	?*
dtype0
?
gru_15/gru_cell_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_namegru_15/gru_cell_15/kernel
?
-gru_15/gru_cell_15/kernel/Read/ReadVariableOpReadVariableOpgru_15/gru_cell_15/kernel* 
_output_shapes
:
??*
dtype0
?
#gru_15/gru_cell_15/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#gru_15/gru_cell_15/recurrent_kernel
?
7gru_15/gru_cell_15/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_15/gru_cell_15/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
gru_15/gru_cell_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_namegru_15/gru_cell_15/bias
?
+gru_15/gru_cell_15/bias/Read/ReadVariableOpReadVariableOpgru_15/gru_cell_15/bias*
_output_shapes
:	?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_14/kernel/m
?
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_14/bias/m
z
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_15/kernel/m
?
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:*
dtype0
?
 Adam/gru_14/gru_cell_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/gru_14/gru_cell_14/kernel/m
?
4Adam/gru_14/gru_cell_14/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_14/gru_cell_14/kernel/m*
_output_shapes
:	?*
dtype0
?
*Adam/gru_14/gru_cell_14/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_14/gru_cell_14/recurrent_kernel/m
?
>Adam/gru_14/gru_cell_14/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_14/gru_cell_14/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/gru_14/gru_cell_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_14/gru_cell_14/bias/m
?
2Adam/gru_14/gru_cell_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_14/gru_cell_14/bias/m*
_output_shapes
:	?*
dtype0
?
 Adam/gru_15/gru_cell_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/gru_15/gru_cell_15/kernel/m
?
4Adam/gru_15/gru_cell_15/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_15/gru_cell_15/kernel/m* 
_output_shapes
:
??*
dtype0
?
*Adam/gru_15/gru_cell_15/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_15/gru_cell_15/recurrent_kernel/m
?
>Adam/gru_15/gru_cell_15/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_15/gru_cell_15/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/gru_15/gru_cell_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_15/gru_cell_15/bias/m
?
2Adam/gru_15/gru_cell_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_15/gru_cell_15/bias/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_14/kernel/v
?
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_14/bias/v
z
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_15/kernel/v
?
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:*
dtype0
?
 Adam/gru_14/gru_cell_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/gru_14/gru_cell_14/kernel/v
?
4Adam/gru_14/gru_cell_14/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_14/gru_cell_14/kernel/v*
_output_shapes
:	?*
dtype0
?
*Adam/gru_14/gru_cell_14/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_14/gru_cell_14/recurrent_kernel/v
?
>Adam/gru_14/gru_cell_14/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_14/gru_cell_14/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/gru_14/gru_cell_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_14/gru_cell_14/bias/v
?
2Adam/gru_14/gru_cell_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_14/gru_cell_14/bias/v*
_output_shapes
:	?*
dtype0
?
 Adam/gru_15/gru_cell_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/gru_15/gru_cell_15/kernel/v
?
4Adam/gru_15/gru_cell_15/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_15/gru_cell_15/kernel/v* 
_output_shapes
:
??*
dtype0
?
*Adam/gru_15/gru_cell_15/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_15/gru_cell_15/recurrent_kernel/v
?
>Adam/gru_15/gru_cell_15/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_15/gru_cell_15/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/gru_15/gru_cell_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_15/gru_cell_15/bias/v
?
2Adam/gru_15/gru_cell_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_15/gru_cell_15/bias/v*
_output_shapes
:	?*
dtype0

NoOpNoOp
?B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?B
value?BB?B B?A
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
h

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
?
2iter

3beta_1

4beta_2
	5decay
6learning_rate"m?#m?,m?-m?7m?8m?9m?:m?;m?<m?"v?#v?,v?-v?7v?8v?9v?:v?;v?<v?
F
70
81
92
:3
;4
<5
"6
#7
,8
-9
 
F
70
81
92
:3
;4
<5
"6
#7
,8
-9
?
	trainable_variables

regularization_losses
=layer_regularization_losses

>layers
?layer_metrics
	variables
@non_trainable_variables
Ametrics
 
~

7kernel
8recurrent_kernel
9bias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
 

70
81
92
 

70
81
92
?
trainable_variables

Fstates
regularization_losses
Glayer_regularization_losses

Hlayers
Ilayer_metrics
	variables
Jnon_trainable_variables
Kmetrics
 
 
 
?
trainable_variables
regularization_losses
	variables
Llayer_regularization_losses

Mlayers
Nlayer_metrics
Onon_trainable_variables
Pmetrics
~

:kernel
;recurrent_kernel
<bias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
 

:0
;1
<2
 

:0
;1
<2
?
trainable_variables

Ustates
regularization_losses
Vlayer_regularization_losses

Wlayers
Xlayer_metrics
	variables
Ynon_trainable_variables
Zmetrics
 
 
 
?
trainable_variables
regularization_losses
 	variables
[layer_regularization_losses

\layers
]layer_metrics
^non_trainable_variables
_metrics
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
?
$trainable_variables
%regularization_losses
&	variables
`layer_regularization_losses

alayers
blayer_metrics
cnon_trainable_variables
dmetrics
 
 
 
?
(trainable_variables
)regularization_losses
*	variables
elayer_regularization_losses

flayers
glayer_metrics
hnon_trainable_variables
imetrics
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
?
.trainable_variables
/regularization_losses
0	variables
jlayer_regularization_losses

klayers
llayer_metrics
mnon_trainable_variables
nmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEgru_14/gru_cell_14/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#gru_14/gru_cell_14/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_14/gru_cell_14/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEgru_15/gru_cell_15/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#gru_15/gru_cell_15/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_15/gru_cell_15/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
1
0
1
2
3
4
5
6
 
 

o0
p1

70
81
92
 

70
81
92
?
Btrainable_variables
Cregularization_losses
D	variables
qlayer_regularization_losses

rlayers
slayer_metrics
tnon_trainable_variables
umetrics
 
 

0
 
 
 
 
 
 
 
 

:0
;1
<2
 

:0
;1
<2
?
Qtrainable_variables
Rregularization_losses
S	variables
vlayer_regularization_losses

wlayers
xlayer_metrics
ynon_trainable_variables
zmetrics
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	{total
	|count
}	variables
~	keras_api
H
	total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

{0
|1

}	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
?1

?	variables
~|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/gru_14/gru_cell_14/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_14/gru_cell_14/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/gru_14/gru_cell_14/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/gru_15/gru_cell_15/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_15/gru_cell_15/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/gru_15/gru_cell_15/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/gru_14/gru_cell_14/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_14/gru_cell_14/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/gru_14/gru_cell_14/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/gru_15/gru_cell_15/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_15/gru_cell_15/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/gru_15/gru_cell_15/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_gru_14_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_14_inputgru_14/gru_cell_14/biasgru_14/gru_cell_14/kernel#gru_14/gru_cell_14/recurrent_kernelgru_15/gru_cell_15/biasgru_15/gru_cell_15/kernel#gru_15/gru_cell_15/recurrent_kerneldense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_699694
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-gru_14/gru_cell_14/kernel/Read/ReadVariableOp7gru_14/gru_cell_14/recurrent_kernel/Read/ReadVariableOp+gru_14/gru_cell_14/bias/Read/ReadVariableOp-gru_15/gru_cell_15/kernel/Read/ReadVariableOp7gru_15/gru_cell_15/recurrent_kernel/Read/ReadVariableOp+gru_15/gru_cell_15/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp4Adam/gru_14/gru_cell_14/kernel/m/Read/ReadVariableOp>Adam/gru_14/gru_cell_14/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_14/gru_cell_14/bias/m/Read/ReadVariableOp4Adam/gru_15/gru_cell_15/kernel/m/Read/ReadVariableOp>Adam/gru_15/gru_cell_15/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_15/gru_cell_15/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOp4Adam/gru_14/gru_cell_14/kernel/v/Read/ReadVariableOp>Adam/gru_14/gru_cell_14/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_14/gru_cell_14/bias/v/Read/ReadVariableOp4Adam/gru_15/gru_cell_15/kernel/v/Read/ReadVariableOp>Adam/gru_15/gru_cell_15/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_15/gru_cell_15/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_702305
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/biasdense_15/kerneldense_15/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_14/gru_cell_14/kernel#gru_14/gru_cell_14/recurrent_kernelgru_14/gru_cell_14/biasgru_15/gru_cell_15/kernel#gru_15/gru_cell_15/recurrent_kernelgru_15/gru_cell_15/biastotalcounttotal_1count_1Adam/dense_14/kernel/mAdam/dense_14/bias/mAdam/dense_15/kernel/mAdam/dense_15/bias/m Adam/gru_14/gru_cell_14/kernel/m*Adam/gru_14/gru_cell_14/recurrent_kernel/mAdam/gru_14/gru_cell_14/bias/m Adam/gru_15/gru_cell_15/kernel/m*Adam/gru_15/gru_cell_15/recurrent_kernel/mAdam/gru_15/gru_cell_15/bias/mAdam/dense_14/kernel/vAdam/dense_14/bias/vAdam/dense_15/kernel/vAdam/dense_15/bias/v Adam/gru_14/gru_cell_14/kernel/v*Adam/gru_14/gru_cell_14/recurrent_kernel/vAdam/gru_14/gru_cell_14/bias/v Adam/gru_15/gru_cell_15/kernel/v*Adam/gru_15/gru_cell_15/recurrent_kernel/vAdam/gru_15/gru_cell_15/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_702432??'
?X
?
B__inference_gru_15_layer_call_and_return_conditional_losses_701776

inputs6
#gru_cell_15_readvariableop_resource:	?>
*gru_cell_15_matmul_readvariableop_resource:
??@
,gru_cell_15_matmul_1_readvariableop_resource:
??
identity??!gru_cell_15/MatMul/ReadVariableOp?#gru_cell_15/MatMul_1/ReadVariableOp?gru_cell_15/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_15/ReadVariableOpReadVariableOp#gru_cell_15_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_15/ReadVariableOp?
gru_cell_15/unstackUnpack"gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_15/unstack?
!gru_cell_15/MatMul/ReadVariableOpReadVariableOp*gru_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_15/MatMul/ReadVariableOp?
gru_cell_15/MatMulMatMulstrided_slice_2:output:0)gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul?
gru_cell_15/BiasAddBiasAddgru_cell_15/MatMul:product:0gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd?
gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split/split_dim?
gru_cell_15/splitSplit$gru_cell_15/split/split_dim:output:0gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split?
#gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_15/MatMul_1/ReadVariableOp?
gru_cell_15/MatMul_1MatMulzeros:output:0+gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul_1?
gru_cell_15/BiasAdd_1BiasAddgru_cell_15/MatMul_1:product:0gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd_1{
gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_15/Const?
gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split_1/split_dim?
gru_cell_15/split_1SplitVgru_cell_15/BiasAdd_1:output:0gru_cell_15/Const:output:0&gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split_1?
gru_cell_15/addAddV2gru_cell_15/split:output:0gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add}
gru_cell_15/SigmoidSigmoidgru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid?
gru_cell_15/add_1AddV2gru_cell_15/split:output:1gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_1?
gru_cell_15/Sigmoid_1Sigmoidgru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid_1?
gru_cell_15/mulMulgru_cell_15/Sigmoid_1:y:0gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul?
gru_cell_15/add_2AddV2gru_cell_15/split:output:2gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_2v
gru_cell_15/ReluRelugru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Relu?
gru_cell_15/mul_1Mulgru_cell_15/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_1k
gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_15/sub/x?
gru_cell_15/subSubgru_cell_15/sub/x:output:0gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/sub?
gru_cell_15/mul_2Mulgru_cell_15/sub:z:0gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_2?
gru_cell_15/add_3AddV2gru_cell_15/mul_1:z:0gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_15_readvariableop_resource*gru_cell_15_matmul_readvariableop_resource,gru_cell_15_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_701687*
condR
while_cond_701686*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp"^gru_cell_15/MatMul/ReadVariableOp$^gru_cell_15/MatMul_1/ReadVariableOp^gru_cell_15/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_15/MatMul/ReadVariableOp!gru_cell_15/MatMul/ReadVariableOp2J
#gru_cell_15/MatMul_1/ReadVariableOp#gru_cell_15/MatMul_1/ReadVariableOp28
gru_cell_15/ReadVariableOpgru_cell_15/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
gru_14_while_cond_699757*
&gru_14_while_gru_14_while_loop_counter0
,gru_14_while_gru_14_while_maximum_iterations
gru_14_while_placeholder
gru_14_while_placeholder_1
gru_14_while_placeholder_2,
(gru_14_while_less_gru_14_strided_slice_1B
>gru_14_while_gru_14_while_cond_699757___redundant_placeholder0B
>gru_14_while_gru_14_while_cond_699757___redundant_placeholder1B
>gru_14_while_gru_14_while_cond_699757___redundant_placeholder2B
>gru_14_while_gru_14_while_cond_699757___redundant_placeholder3
gru_14_while_identity
?
gru_14/while/LessLessgru_14_while_placeholder(gru_14_while_less_gru_14_strided_slice_1*
T0*
_output_shapes
: 2
gru_14/while/Lessr
gru_14/while/IdentityIdentitygru_14/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_14/while/Identity"7
gru_14_while_identitygru_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
-__inference_sequential_7_layer_call_fn_700456

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_6990312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_698845
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_698845___redundant_placeholder04
0while_while_cond_698845___redundant_placeholder14
0while_while_cond_698845___redundant_placeholder24
0while_while_cond_698845___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_700697
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_700697___redundant_placeholder04
0while_while_cond_700697___redundant_placeholder14
0while_while_cond_700697___redundant_placeholder24
0while_while_cond_700697___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_701003
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_701003___redundant_placeholder04
0while_while_cond_701003___redundant_placeholder14
0while_while_cond_701003___redundant_placeholder24
0while_while_cond_701003___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
,__inference_gru_cell_15_layer_call_fn_702165

inputs
states_0
unknown:	?
	unknown_0:
??
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_6982562
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
d
+__inference_dropout_21_layer_call_fn_701164

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_6993152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_gru_14_layer_call_fn_701104
inputs_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_14_layer_call_and_return_conditional_losses_6976242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?;
?
B__inference_gru_14_layer_call_and_return_conditional_losses_697624

inputs%
gru_cell_14_697548:	?%
gru_cell_14_697550:	?&
gru_cell_14_697552:
??
identity??#gru_cell_14/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_14_697548gru_cell_14_697550gru_cell_14_697552*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_6975472%
#gru_cell_14/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_14_697548gru_cell_14_697550gru_cell_14_697552*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_697560*
condR
while_cond_697559*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity|
NoOpNoOp$^gru_cell_14/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#gru_cell_14/StatefulPartitionedCall#gru_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?d
?
%sequential_7_gru_15_while_body_697333D
@sequential_7_gru_15_while_sequential_7_gru_15_while_loop_counterJ
Fsequential_7_gru_15_while_sequential_7_gru_15_while_maximum_iterations)
%sequential_7_gru_15_while_placeholder+
'sequential_7_gru_15_while_placeholder_1+
'sequential_7_gru_15_while_placeholder_2C
?sequential_7_gru_15_while_sequential_7_gru_15_strided_slice_1_0
{sequential_7_gru_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_gru_15_tensorarrayunstack_tensorlistfromtensor_0R
?sequential_7_gru_15_while_gru_cell_15_readvariableop_resource_0:	?Z
Fsequential_7_gru_15_while_gru_cell_15_matmul_readvariableop_resource_0:
??\
Hsequential_7_gru_15_while_gru_cell_15_matmul_1_readvariableop_resource_0:
??&
"sequential_7_gru_15_while_identity(
$sequential_7_gru_15_while_identity_1(
$sequential_7_gru_15_while_identity_2(
$sequential_7_gru_15_while_identity_3(
$sequential_7_gru_15_while_identity_4A
=sequential_7_gru_15_while_sequential_7_gru_15_strided_slice_1}
ysequential_7_gru_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_gru_15_tensorarrayunstack_tensorlistfromtensorP
=sequential_7_gru_15_while_gru_cell_15_readvariableop_resource:	?X
Dsequential_7_gru_15_while_gru_cell_15_matmul_readvariableop_resource:
??Z
Fsequential_7_gru_15_while_gru_cell_15_matmul_1_readvariableop_resource:
????;sequential_7/gru_15/while/gru_cell_15/MatMul/ReadVariableOp?=sequential_7/gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp?4sequential_7/gru_15/while/gru_cell_15/ReadVariableOp?
Ksequential_7/gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Ksequential_7/gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_7/gru_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_7_gru_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_gru_15_tensorarrayunstack_tensorlistfromtensor_0%sequential_7_gru_15_while_placeholderTsequential_7/gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02?
=sequential_7/gru_15/while/TensorArrayV2Read/TensorListGetItem?
4sequential_7/gru_15/while/gru_cell_15/ReadVariableOpReadVariableOp?sequential_7_gru_15_while_gru_cell_15_readvariableop_resource_0*
_output_shapes
:	?*
dtype026
4sequential_7/gru_15/while/gru_cell_15/ReadVariableOp?
-sequential_7/gru_15/while/gru_cell_15/unstackUnpack<sequential_7/gru_15/while/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2/
-sequential_7/gru_15/while/gru_cell_15/unstack?
;sequential_7/gru_15/while/gru_cell_15/MatMul/ReadVariableOpReadVariableOpFsequential_7_gru_15_while_gru_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02=
;sequential_7/gru_15/while/gru_cell_15/MatMul/ReadVariableOp?
,sequential_7/gru_15/while/gru_cell_15/MatMulMatMulDsequential_7/gru_15/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_7/gru_15/while/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,sequential_7/gru_15/while/gru_cell_15/MatMul?
-sequential_7/gru_15/while/gru_cell_15/BiasAddBiasAdd6sequential_7/gru_15/while/gru_cell_15/MatMul:product:06sequential_7/gru_15/while/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2/
-sequential_7/gru_15/while/gru_cell_15/BiasAdd?
5sequential_7/gru_15/while/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5sequential_7/gru_15/while/gru_cell_15/split/split_dim?
+sequential_7/gru_15/while/gru_cell_15/splitSplit>sequential_7/gru_15/while/gru_cell_15/split/split_dim:output:06sequential_7/gru_15/while/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2-
+sequential_7/gru_15/while/gru_cell_15/split?
=sequential_7/gru_15/while/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOpHsequential_7_gru_15_while_gru_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02?
=sequential_7/gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp?
.sequential_7/gru_15/while/gru_cell_15/MatMul_1MatMul'sequential_7_gru_15_while_placeholder_2Esequential_7/gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_7/gru_15/while/gru_cell_15/MatMul_1?
/sequential_7/gru_15/while/gru_cell_15/BiasAdd_1BiasAdd8sequential_7/gru_15/while/gru_cell_15/MatMul_1:product:06sequential_7/gru_15/while/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????21
/sequential_7/gru_15/while/gru_cell_15/BiasAdd_1?
+sequential_7/gru_15/while/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2-
+sequential_7/gru_15/while/gru_cell_15/Const?
7sequential_7/gru_15/while/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????29
7sequential_7/gru_15/while/gru_cell_15/split_1/split_dim?
-sequential_7/gru_15/while/gru_cell_15/split_1SplitV8sequential_7/gru_15/while/gru_cell_15/BiasAdd_1:output:04sequential_7/gru_15/while/gru_cell_15/Const:output:0@sequential_7/gru_15/while/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2/
-sequential_7/gru_15/while/gru_cell_15/split_1?
)sequential_7/gru_15/while/gru_cell_15/addAddV24sequential_7/gru_15/while/gru_cell_15/split:output:06sequential_7/gru_15/while/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_7/gru_15/while/gru_cell_15/add?
-sequential_7/gru_15/while/gru_cell_15/SigmoidSigmoid-sequential_7/gru_15/while/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2/
-sequential_7/gru_15/while/gru_cell_15/Sigmoid?
+sequential_7/gru_15/while/gru_cell_15/add_1AddV24sequential_7/gru_15/while/gru_cell_15/split:output:16sequential_7/gru_15/while/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2-
+sequential_7/gru_15/while/gru_cell_15/add_1?
/sequential_7/gru_15/while/gru_cell_15/Sigmoid_1Sigmoid/sequential_7/gru_15/while/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????21
/sequential_7/gru_15/while/gru_cell_15/Sigmoid_1?
)sequential_7/gru_15/while/gru_cell_15/mulMul3sequential_7/gru_15/while/gru_cell_15/Sigmoid_1:y:06sequential_7/gru_15/while/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2+
)sequential_7/gru_15/while/gru_cell_15/mul?
+sequential_7/gru_15/while/gru_cell_15/add_2AddV24sequential_7/gru_15/while/gru_cell_15/split:output:2-sequential_7/gru_15/while/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2-
+sequential_7/gru_15/while/gru_cell_15/add_2?
*sequential_7/gru_15/while/gru_cell_15/ReluRelu/sequential_7/gru_15/while/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2,
*sequential_7/gru_15/while/gru_cell_15/Relu?
+sequential_7/gru_15/while/gru_cell_15/mul_1Mul1sequential_7/gru_15/while/gru_cell_15/Sigmoid:y:0'sequential_7_gru_15_while_placeholder_2*
T0*(
_output_shapes
:??????????2-
+sequential_7/gru_15/while/gru_cell_15/mul_1?
+sequential_7/gru_15/while/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+sequential_7/gru_15/while/gru_cell_15/sub/x?
)sequential_7/gru_15/while/gru_cell_15/subSub4sequential_7/gru_15/while/gru_cell_15/sub/x:output:01sequential_7/gru_15/while/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2+
)sequential_7/gru_15/while/gru_cell_15/sub?
+sequential_7/gru_15/while/gru_cell_15/mul_2Mul-sequential_7/gru_15/while/gru_cell_15/sub:z:08sequential_7/gru_15/while/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2-
+sequential_7/gru_15/while/gru_cell_15/mul_2?
+sequential_7/gru_15/while/gru_cell_15/add_3AddV2/sequential_7/gru_15/while/gru_cell_15/mul_1:z:0/sequential_7/gru_15/while/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2-
+sequential_7/gru_15/while/gru_cell_15/add_3?
>sequential_7/gru_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_7_gru_15_while_placeholder_1%sequential_7_gru_15_while_placeholder/sequential_7/gru_15/while/gru_cell_15/add_3:z:0*
_output_shapes
: *
element_dtype02@
>sequential_7/gru_15/while/TensorArrayV2Write/TensorListSetItem?
sequential_7/gru_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_7/gru_15/while/add/y?
sequential_7/gru_15/while/addAddV2%sequential_7_gru_15_while_placeholder(sequential_7/gru_15/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_7/gru_15/while/add?
!sequential_7/gru_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_7/gru_15/while/add_1/y?
sequential_7/gru_15/while/add_1AddV2@sequential_7_gru_15_while_sequential_7_gru_15_while_loop_counter*sequential_7/gru_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_7/gru_15/while/add_1?
"sequential_7/gru_15/while/IdentityIdentity#sequential_7/gru_15/while/add_1:z:0^sequential_7/gru_15/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_7/gru_15/while/Identity?
$sequential_7/gru_15/while/Identity_1IdentityFsequential_7_gru_15_while_sequential_7_gru_15_while_maximum_iterations^sequential_7/gru_15/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_7/gru_15/while/Identity_1?
$sequential_7/gru_15/while/Identity_2Identity!sequential_7/gru_15/while/add:z:0^sequential_7/gru_15/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_7/gru_15/while/Identity_2?
$sequential_7/gru_15/while/Identity_3IdentityNsequential_7/gru_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_7/gru_15/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_7/gru_15/while/Identity_3?
$sequential_7/gru_15/while/Identity_4Identity/sequential_7/gru_15/while/gru_cell_15/add_3:z:0^sequential_7/gru_15/while/NoOp*
T0*(
_output_shapes
:??????????2&
$sequential_7/gru_15/while/Identity_4?
sequential_7/gru_15/while/NoOpNoOp<^sequential_7/gru_15/while/gru_cell_15/MatMul/ReadVariableOp>^sequential_7/gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp5^sequential_7/gru_15/while/gru_cell_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_7/gru_15/while/NoOp"?
Fsequential_7_gru_15_while_gru_cell_15_matmul_1_readvariableop_resourceHsequential_7_gru_15_while_gru_cell_15_matmul_1_readvariableop_resource_0"?
Dsequential_7_gru_15_while_gru_cell_15_matmul_readvariableop_resourceFsequential_7_gru_15_while_gru_cell_15_matmul_readvariableop_resource_0"?
=sequential_7_gru_15_while_gru_cell_15_readvariableop_resource?sequential_7_gru_15_while_gru_cell_15_readvariableop_resource_0"Q
"sequential_7_gru_15_while_identity+sequential_7/gru_15/while/Identity:output:0"U
$sequential_7_gru_15_while_identity_1-sequential_7/gru_15/while/Identity_1:output:0"U
$sequential_7_gru_15_while_identity_2-sequential_7/gru_15/while/Identity_2:output:0"U
$sequential_7_gru_15_while_identity_3-sequential_7/gru_15/while/Identity_3:output:0"U
$sequential_7_gru_15_while_identity_4-sequential_7/gru_15/while/Identity_4:output:0"?
=sequential_7_gru_15_while_sequential_7_gru_15_strided_slice_1?sequential_7_gru_15_while_sequential_7_gru_15_strided_slice_1_0"?
ysequential_7_gru_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_gru_15_tensorarrayunstack_tensorlistfromtensor{sequential_7_gru_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_gru_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2z
;sequential_7/gru_15/while/gru_cell_15/MatMul/ReadVariableOp;sequential_7/gru_15/while/gru_cell_15/MatMul/ReadVariableOp2~
=sequential_7/gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp=sequential_7/gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp2l
4sequential_7/gru_15/while/gru_cell_15/ReadVariableOp4sequential_7/gru_15/while/gru_cell_15/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?E
?
while_body_700545
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_14_readvariableop_resource_0:	?E
2while_gru_cell_14_matmul_readvariableop_resource_0:	?H
4while_gru_cell_14_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_14_readvariableop_resource:	?C
0while_gru_cell_14_matmul_readvariableop_resource:	?F
2while_gru_cell_14_matmul_1_readvariableop_resource:
????'while/gru_cell_14/MatMul/ReadVariableOp?)while/gru_cell_14/MatMul_1/ReadVariableOp? while/gru_cell_14/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_14/ReadVariableOpReadVariableOp+while_gru_cell_14_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_14/ReadVariableOp?
while/gru_cell_14/unstackUnpack(while/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_14/unstack?
'while/gru_cell_14/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_14/MatMul/ReadVariableOp?
while/gru_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul?
while/gru_cell_14/BiasAddBiasAdd"while/gru_cell_14/MatMul:product:0"while/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd?
!while/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_14/split/split_dim?
while/gru_cell_14/splitSplit*while/gru_cell_14/split/split_dim:output:0"while/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split?
)while/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_14/MatMul_1/ReadVariableOp?
while/gru_cell_14/MatMul_1MatMulwhile_placeholder_21while/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul_1?
while/gru_cell_14/BiasAdd_1BiasAdd$while/gru_cell_14/MatMul_1:product:0"while/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd_1?
while/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_14/Const?
#while/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_14/split_1/split_dim?
while/gru_cell_14/split_1SplitV$while/gru_cell_14/BiasAdd_1:output:0 while/gru_cell_14/Const:output:0,while/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split_1?
while/gru_cell_14/addAddV2 while/gru_cell_14/split:output:0"while/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add?
while/gru_cell_14/SigmoidSigmoidwhile/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid?
while/gru_cell_14/add_1AddV2 while/gru_cell_14/split:output:1"while/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_1?
while/gru_cell_14/Sigmoid_1Sigmoidwhile/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid_1?
while/gru_cell_14/mulMulwhile/gru_cell_14/Sigmoid_1:y:0"while/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul?
while/gru_cell_14/add_2AddV2 while/gru_cell_14/split:output:2while/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_2?
while/gru_cell_14/ReluReluwhile/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Relu?
while/gru_cell_14/mul_1Mulwhile/gru_cell_14/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_1w
while/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_14/sub/x?
while/gru_cell_14/subSub while/gru_cell_14/sub/x:output:0while/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/sub?
while/gru_cell_14/mul_2Mulwhile/gru_cell_14/sub:z:0$while/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_2?
while/gru_cell_14/add_3AddV2while/gru_cell_14/mul_1:z:0while/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_14/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_14/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_14/MatMul/ReadVariableOp*^while/gru_cell_14/MatMul_1/ReadVariableOp!^while/gru_cell_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_14_matmul_1_readvariableop_resource4while_gru_cell_14_matmul_1_readvariableop_resource_0"f
0while_gru_cell_14_matmul_readvariableop_resource2while_gru_cell_14_matmul_readvariableop_resource_0"X
)while_gru_cell_14_readvariableop_resource+while_gru_cell_14_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_14/MatMul/ReadVariableOp'while/gru_cell_14/MatMul/ReadVariableOp2V
)while/gru_cell_14/MatMul_1/ReadVariableOp)while/gru_cell_14/MatMul_1/ReadVariableOp2D
 while/gru_cell_14/ReadVariableOp while/gru_cell_14/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_698678
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_698678___redundant_placeholder04
0while_while_cond_698678___redundant_placeholder14
0while_while_cond_698678___redundant_placeholder24
0while_while_cond_698678___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_698125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_698125___redundant_placeholder04
0while_while_cond_698125___redundant_placeholder14
0while_while_cond_698125___redundant_placeholder24
0while_while_cond_698125___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
G
+__inference_dropout_21_layer_call_fn_701159

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_6987812
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_698948

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_21_layer_call_and_return_conditional_losses_698781

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_699394
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_699394___redundant_placeholder04
0while_while_cond_699394___redundant_placeholder14
0while_while_cond_699394___redundant_placeholder24
0while_while_cond_699394___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
d
+__inference_dropout_23_layer_call_fn_701914

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_6990842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?P
?	
gru_15_while_body_699908*
&gru_15_while_gru_15_while_loop_counter0
,gru_15_while_gru_15_while_maximum_iterations
gru_15_while_placeholder
gru_15_while_placeholder_1
gru_15_while_placeholder_2)
%gru_15_while_gru_15_strided_slice_1_0e
agru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0E
2gru_15_while_gru_cell_15_readvariableop_resource_0:	?M
9gru_15_while_gru_cell_15_matmul_readvariableop_resource_0:
??O
;gru_15_while_gru_cell_15_matmul_1_readvariableop_resource_0:
??
gru_15_while_identity
gru_15_while_identity_1
gru_15_while_identity_2
gru_15_while_identity_3
gru_15_while_identity_4'
#gru_15_while_gru_15_strided_slice_1c
_gru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensorC
0gru_15_while_gru_cell_15_readvariableop_resource:	?K
7gru_15_while_gru_cell_15_matmul_readvariableop_resource:
??M
9gru_15_while_gru_cell_15_matmul_1_readvariableop_resource:
????.gru_15/while/gru_cell_15/MatMul/ReadVariableOp?0gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp?'gru_15/while/gru_cell_15/ReadVariableOp?
>gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0gru_15_while_placeholderGgru_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0gru_15/while/TensorArrayV2Read/TensorListGetItem?
'gru_15/while/gru_cell_15/ReadVariableOpReadVariableOp2gru_15_while_gru_cell_15_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_15/while/gru_cell_15/ReadVariableOp?
 gru_15/while/gru_cell_15/unstackUnpack/gru_15/while/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_15/while/gru_cell_15/unstack?
.gru_15/while/gru_cell_15/MatMul/ReadVariableOpReadVariableOp9gru_15_while_gru_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype020
.gru_15/while/gru_cell_15/MatMul/ReadVariableOp?
gru_15/while/gru_cell_15/MatMulMatMul7gru_15/while/TensorArrayV2Read/TensorListGetItem:item:06gru_15/while/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_15/while/gru_cell_15/MatMul?
 gru_15/while/gru_cell_15/BiasAddBiasAdd)gru_15/while/gru_cell_15/MatMul:product:0)gru_15/while/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_15/while/gru_cell_15/BiasAdd?
(gru_15/while/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_15/while/gru_cell_15/split/split_dim?
gru_15/while/gru_cell_15/splitSplit1gru_15/while/gru_cell_15/split/split_dim:output:0)gru_15/while/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_15/while/gru_cell_15/split?
0gru_15/while/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp;gru_15_while_gru_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp?
!gru_15/while/gru_cell_15/MatMul_1MatMulgru_15_while_placeholder_28gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_15/while/gru_cell_15/MatMul_1?
"gru_15/while/gru_cell_15/BiasAdd_1BiasAdd+gru_15/while/gru_cell_15/MatMul_1:product:0)gru_15/while/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_15/while/gru_cell_15/BiasAdd_1?
gru_15/while/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2 
gru_15/while/gru_cell_15/Const?
*gru_15/while/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_15/while/gru_cell_15/split_1/split_dim?
 gru_15/while/gru_cell_15/split_1SplitV+gru_15/while/gru_cell_15/BiasAdd_1:output:0'gru_15/while/gru_cell_15/Const:output:03gru_15/while/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_15/while/gru_cell_15/split_1?
gru_15/while/gru_cell_15/addAddV2'gru_15/while/gru_cell_15/split:output:0)gru_15/while/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_15/while/gru_cell_15/add?
 gru_15/while/gru_cell_15/SigmoidSigmoid gru_15/while/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_15/while/gru_cell_15/Sigmoid?
gru_15/while/gru_cell_15/add_1AddV2'gru_15/while/gru_cell_15/split:output:1)gru_15/while/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_15/while/gru_cell_15/add_1?
"gru_15/while/gru_cell_15/Sigmoid_1Sigmoid"gru_15/while/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_15/while/gru_cell_15/Sigmoid_1?
gru_15/while/gru_cell_15/mulMul&gru_15/while/gru_cell_15/Sigmoid_1:y:0)gru_15/while/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_15/while/gru_cell_15/mul?
gru_15/while/gru_cell_15/add_2AddV2'gru_15/while/gru_cell_15/split:output:2 gru_15/while/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_15/while/gru_cell_15/add_2?
gru_15/while/gru_cell_15/ReluRelu"gru_15/while/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_15/while/gru_cell_15/Relu?
gru_15/while/gru_cell_15/mul_1Mul$gru_15/while/gru_cell_15/Sigmoid:y:0gru_15_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_15/while/gru_cell_15/mul_1?
gru_15/while/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_15/while/gru_cell_15/sub/x?
gru_15/while/gru_cell_15/subSub'gru_15/while/gru_cell_15/sub/x:output:0$gru_15/while/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_15/while/gru_cell_15/sub?
gru_15/while/gru_cell_15/mul_2Mul gru_15/while/gru_cell_15/sub:z:0+gru_15/while/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_15/while/gru_cell_15/mul_2?
gru_15/while/gru_cell_15/add_3AddV2"gru_15/while/gru_cell_15/mul_1:z:0"gru_15/while/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_15/while/gru_cell_15/add_3?
1gru_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_15_while_placeholder_1gru_15_while_placeholder"gru_15/while/gru_cell_15/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_15/while/TensorArrayV2Write/TensorListSetItemj
gru_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_15/while/add/y?
gru_15/while/addAddV2gru_15_while_placeholdergru_15/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_15/while/addn
gru_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_15/while/add_1/y?
gru_15/while/add_1AddV2&gru_15_while_gru_15_while_loop_countergru_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_15/while/add_1?
gru_15/while/IdentityIdentitygru_15/while/add_1:z:0^gru_15/while/NoOp*
T0*
_output_shapes
: 2
gru_15/while/Identity?
gru_15/while/Identity_1Identity,gru_15_while_gru_15_while_maximum_iterations^gru_15/while/NoOp*
T0*
_output_shapes
: 2
gru_15/while/Identity_1?
gru_15/while/Identity_2Identitygru_15/while/add:z:0^gru_15/while/NoOp*
T0*
_output_shapes
: 2
gru_15/while/Identity_2?
gru_15/while/Identity_3IdentityAgru_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_15/while/NoOp*
T0*
_output_shapes
: 2
gru_15/while/Identity_3?
gru_15/while/Identity_4Identity"gru_15/while/gru_cell_15/add_3:z:0^gru_15/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_15/while/Identity_4?
gru_15/while/NoOpNoOp/^gru_15/while/gru_cell_15/MatMul/ReadVariableOp1^gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp(^gru_15/while/gru_cell_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_15/while/NoOp"L
#gru_15_while_gru_15_strided_slice_1%gru_15_while_gru_15_strided_slice_1_0"x
9gru_15_while_gru_cell_15_matmul_1_readvariableop_resource;gru_15_while_gru_cell_15_matmul_1_readvariableop_resource_0"t
7gru_15_while_gru_cell_15_matmul_readvariableop_resource9gru_15_while_gru_cell_15_matmul_readvariableop_resource_0"f
0gru_15_while_gru_cell_15_readvariableop_resource2gru_15_while_gru_cell_15_readvariableop_resource_0"7
gru_15_while_identitygru_15/while/Identity:output:0";
gru_15_while_identity_1 gru_15/while/Identity_1:output:0";
gru_15_while_identity_2 gru_15/while/Identity_2:output:0";
gru_15_while_identity_3 gru_15/while/Identity_3:output:0";
gru_15_while_identity_4 gru_15/while/Identity_4:output:0"?
_gru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensoragru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_15/while/gru_cell_15/MatMul/ReadVariableOp.gru_15/while/gru_cell_15/MatMul/ReadVariableOp2d
0gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp0gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp2R
'gru_15/while/gru_cell_15/ReadVariableOp'gru_15/while/gru_cell_15/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?E
?
while_body_699197
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_15_readvariableop_resource_0:	?F
2while_gru_cell_15_matmul_readvariableop_resource_0:
??H
4while_gru_cell_15_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_15_readvariableop_resource:	?D
0while_gru_cell_15_matmul_readvariableop_resource:
??F
2while_gru_cell_15_matmul_1_readvariableop_resource:
????'while/gru_cell_15/MatMul/ReadVariableOp?)while/gru_cell_15/MatMul_1/ReadVariableOp? while/gru_cell_15/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_15/ReadVariableOpReadVariableOp+while_gru_cell_15_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_15/ReadVariableOp?
while/gru_cell_15/unstackUnpack(while/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_15/unstack?
'while/gru_cell_15/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_15/MatMul/ReadVariableOp?
while/gru_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul?
while/gru_cell_15/BiasAddBiasAdd"while/gru_cell_15/MatMul:product:0"while/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd?
!while/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_15/split/split_dim?
while/gru_cell_15/splitSplit*while/gru_cell_15/split/split_dim:output:0"while/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split?
)while/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_15/MatMul_1/ReadVariableOp?
while/gru_cell_15/MatMul_1MatMulwhile_placeholder_21while/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul_1?
while/gru_cell_15/BiasAdd_1BiasAdd$while/gru_cell_15/MatMul_1:product:0"while/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd_1?
while/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_15/Const?
#while/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_15/split_1/split_dim?
while/gru_cell_15/split_1SplitV$while/gru_cell_15/BiasAdd_1:output:0 while/gru_cell_15/Const:output:0,while/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split_1?
while/gru_cell_15/addAddV2 while/gru_cell_15/split:output:0"while/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add?
while/gru_cell_15/SigmoidSigmoidwhile/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid?
while/gru_cell_15/add_1AddV2 while/gru_cell_15/split:output:1"while/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_1?
while/gru_cell_15/Sigmoid_1Sigmoidwhile/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid_1?
while/gru_cell_15/mulMulwhile/gru_cell_15/Sigmoid_1:y:0"while/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul?
while/gru_cell_15/add_2AddV2 while/gru_cell_15/split:output:2while/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_2?
while/gru_cell_15/ReluReluwhile/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Relu?
while/gru_cell_15/mul_1Mulwhile/gru_cell_15/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_1w
while/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_15/sub/x?
while/gru_cell_15/subSub while/gru_cell_15/sub/x:output:0while/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/sub?
while/gru_cell_15/mul_2Mulwhile/gru_cell_15/sub:z:0$while/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_2?
while/gru_cell_15/add_3AddV2while/gru_cell_15/mul_1:z:0while/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_15/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_15/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_15/MatMul/ReadVariableOp*^while/gru_cell_15/MatMul_1/ReadVariableOp!^while/gru_cell_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_15_matmul_1_readvariableop_resource4while_gru_cell_15_matmul_1_readvariableop_resource_0"f
0while_gru_cell_15_matmul_readvariableop_resource2while_gru_cell_15_matmul_readvariableop_resource_0"X
)while_gru_cell_15_readvariableop_resource+while_gru_cell_15_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_15/MatMul/ReadVariableOp'while/gru_cell_15/MatMul/ReadVariableOp2V
)while/gru_cell_15/MatMul_1/ReadVariableOp)while/gru_cell_15/MatMul_1/ReadVariableOp2D
 while/gru_cell_15/ReadVariableOp while/gru_cell_15/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_21_layer_call_and_return_conditional_losses_701154

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_dense_14_layer_call_fn_701887

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_6989812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?X
?
B__inference_gru_14_layer_call_and_return_conditional_losses_699484

inputs6
#gru_cell_14_readvariableop_resource:	?=
*gru_cell_14_matmul_readvariableop_resource:	?@
,gru_cell_14_matmul_1_readvariableop_resource:
??
identity??!gru_cell_14/MatMul/ReadVariableOp?#gru_cell_14/MatMul_1/ReadVariableOp?gru_cell_14/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_14/ReadVariableOpReadVariableOp#gru_cell_14_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_14/ReadVariableOp?
gru_cell_14/unstackUnpack"gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_14/unstack?
!gru_cell_14/MatMul/ReadVariableOpReadVariableOp*gru_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_14/MatMul/ReadVariableOp?
gru_cell_14/MatMulMatMulstrided_slice_2:output:0)gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul?
gru_cell_14/BiasAddBiasAddgru_cell_14/MatMul:product:0gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd?
gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split/split_dim?
gru_cell_14/splitSplit$gru_cell_14/split/split_dim:output:0gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split?
#gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_14/MatMul_1/ReadVariableOp?
gru_cell_14/MatMul_1MatMulzeros:output:0+gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul_1?
gru_cell_14/BiasAdd_1BiasAddgru_cell_14/MatMul_1:product:0gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd_1{
gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_14/Const?
gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split_1/split_dim?
gru_cell_14/split_1SplitVgru_cell_14/BiasAdd_1:output:0gru_cell_14/Const:output:0&gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split_1?
gru_cell_14/addAddV2gru_cell_14/split:output:0gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add}
gru_cell_14/SigmoidSigmoidgru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid?
gru_cell_14/add_1AddV2gru_cell_14/split:output:1gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_1?
gru_cell_14/Sigmoid_1Sigmoidgru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid_1?
gru_cell_14/mulMulgru_cell_14/Sigmoid_1:y:0gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul?
gru_cell_14/add_2AddV2gru_cell_14/split:output:2gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_2v
gru_cell_14/ReluRelugru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Relu?
gru_cell_14/mul_1Mulgru_cell_14/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_1k
gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_14/sub/x?
gru_cell_14/subSubgru_cell_14/sub/x:output:0gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/sub?
gru_cell_14/mul_2Mulgru_cell_14/sub:z:0gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_2?
gru_cell_14/add_3AddV2gru_cell_14/mul_1:z:0gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_14_readvariableop_resource*gru_cell_14_matmul_readvariableop_resource,gru_cell_14_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_699395*
condR
while_cond_699394*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp"^gru_cell_14/MatMul/ReadVariableOp$^gru_cell_14/MatMul_1/ReadVariableOp^gru_cell_14/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_14/MatMul/ReadVariableOp!gru_cell_14/MatMul/ReadVariableOp2J
#gru_cell_14/MatMul_1/ReadVariableOp#gru_cell_14/MatMul_1/ReadVariableOp28
gru_cell_14/ReadVariableOpgru_cell_14/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%sequential_7_gru_14_while_cond_697182D
@sequential_7_gru_14_while_sequential_7_gru_14_while_loop_counterJ
Fsequential_7_gru_14_while_sequential_7_gru_14_while_maximum_iterations)
%sequential_7_gru_14_while_placeholder+
'sequential_7_gru_14_while_placeholder_1+
'sequential_7_gru_14_while_placeholder_2F
Bsequential_7_gru_14_while_less_sequential_7_gru_14_strided_slice_1\
Xsequential_7_gru_14_while_sequential_7_gru_14_while_cond_697182___redundant_placeholder0\
Xsequential_7_gru_14_while_sequential_7_gru_14_while_cond_697182___redundant_placeholder1\
Xsequential_7_gru_14_while_sequential_7_gru_14_while_cond_697182___redundant_placeholder2\
Xsequential_7_gru_14_while_sequential_7_gru_14_while_cond_697182___redundant_placeholder3&
"sequential_7_gru_14_while_identity
?
sequential_7/gru_14/while/LessLess%sequential_7_gru_14_while_placeholderBsequential_7_gru_14_while_less_sequential_7_gru_14_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_7/gru_14/while/Less?
"sequential_7/gru_14/while/IdentityIdentity"sequential_7/gru_14/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_7/gru_14/while/Identity"Q
"sequential_7_gru_14_while_identity+sequential_7/gru_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
gru_14_while_cond_700115*
&gru_14_while_gru_14_while_loop_counter0
,gru_14_while_gru_14_while_maximum_iterations
gru_14_while_placeholder
gru_14_while_placeholder_1
gru_14_while_placeholder_2,
(gru_14_while_less_gru_14_strided_slice_1B
>gru_14_while_gru_14_while_cond_700115___redundant_placeholder0B
>gru_14_while_gru_14_while_cond_700115___redundant_placeholder1B
>gru_14_while_gru_14_while_cond_700115___redundant_placeholder2B
>gru_14_while_gru_14_while_cond_700115___redundant_placeholder3
gru_14_while_identity
?
gru_14/while/LessLessgru_14_while_placeholder(gru_14_while_less_gru_14_strided_slice_1*
T0*
_output_shapes
: 2
gru_14/while/Lessr
gru_14/while/IdentityIdentitygru_14/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_14/while/Identity"7
gru_14_while_identitygru_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
$__inference_signature_wrapper_699694
gru_14_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_6974772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_14_input
ǯ
?	
H__inference_sequential_7_layer_call_and_return_conditional_losses_700431

inputs=
*gru_14_gru_cell_14_readvariableop_resource:	?D
1gru_14_gru_cell_14_matmul_readvariableop_resource:	?G
3gru_14_gru_cell_14_matmul_1_readvariableop_resource:
??=
*gru_15_gru_cell_15_readvariableop_resource:	?E
1gru_15_gru_cell_15_matmul_readvariableop_resource:
??G
3gru_15_gru_cell_15_matmul_1_readvariableop_resource:
??>
*dense_14_tensordot_readvariableop_resource:
??7
(dense_14_biasadd_readvariableop_resource:	?=
*dense_15_tensordot_readvariableop_resource:	?6
(dense_15_biasadd_readvariableop_resource:
identity??dense_14/BiasAdd/ReadVariableOp?!dense_14/Tensordot/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?!dense_15/Tensordot/ReadVariableOp?(gru_14/gru_cell_14/MatMul/ReadVariableOp?*gru_14/gru_cell_14/MatMul_1/ReadVariableOp?!gru_14/gru_cell_14/ReadVariableOp?gru_14/while?(gru_15/gru_cell_15/MatMul/ReadVariableOp?*gru_15/gru_cell_15/MatMul_1/ReadVariableOp?!gru_15/gru_cell_15/ReadVariableOp?gru_15/whileR
gru_14/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_14/Shape?
gru_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_14/strided_slice/stack?
gru_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_14/strided_slice/stack_1?
gru_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_14/strided_slice/stack_2?
gru_14/strided_sliceStridedSlicegru_14/Shape:output:0#gru_14/strided_slice/stack:output:0%gru_14/strided_slice/stack_1:output:0%gru_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_14/strided_sliceq
gru_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_14/zeros/packed/1?
gru_14/zeros/packedPackgru_14/strided_slice:output:0gru_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_14/zeros/packedm
gru_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_14/zeros/Const?
gru_14/zerosFillgru_14/zeros/packed:output:0gru_14/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_14/zeros?
gru_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_14/transpose/perm?
gru_14/transpose	Transposeinputsgru_14/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
gru_14/transposed
gru_14/Shape_1Shapegru_14/transpose:y:0*
T0*
_output_shapes
:2
gru_14/Shape_1?
gru_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_14/strided_slice_1/stack?
gru_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_14/strided_slice_1/stack_1?
gru_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_14/strided_slice_1/stack_2?
gru_14/strided_slice_1StridedSlicegru_14/Shape_1:output:0%gru_14/strided_slice_1/stack:output:0'gru_14/strided_slice_1/stack_1:output:0'gru_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_14/strided_slice_1?
"gru_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_14/TensorArrayV2/element_shape?
gru_14/TensorArrayV2TensorListReserve+gru_14/TensorArrayV2/element_shape:output:0gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_14/TensorArrayV2?
<gru_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_14/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_14/transpose:y:0Egru_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_14/TensorArrayUnstack/TensorListFromTensor?
gru_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_14/strided_slice_2/stack?
gru_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_14/strided_slice_2/stack_1?
gru_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_14/strided_slice_2/stack_2?
gru_14/strided_slice_2StridedSlicegru_14/transpose:y:0%gru_14/strided_slice_2/stack:output:0'gru_14/strided_slice_2/stack_1:output:0'gru_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_14/strided_slice_2?
!gru_14/gru_cell_14/ReadVariableOpReadVariableOp*gru_14_gru_cell_14_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_14/gru_cell_14/ReadVariableOp?
gru_14/gru_cell_14/unstackUnpack)gru_14/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_14/gru_cell_14/unstack?
(gru_14/gru_cell_14/MatMul/ReadVariableOpReadVariableOp1gru_14_gru_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_14/gru_cell_14/MatMul/ReadVariableOp?
gru_14/gru_cell_14/MatMulMatMulgru_14/strided_slice_2:output:00gru_14/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/MatMul?
gru_14/gru_cell_14/BiasAddBiasAdd#gru_14/gru_cell_14/MatMul:product:0#gru_14/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/BiasAdd?
"gru_14/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_14/gru_cell_14/split/split_dim?
gru_14/gru_cell_14/splitSplit+gru_14/gru_cell_14/split/split_dim:output:0#gru_14/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_14/gru_cell_14/split?
*gru_14/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp3gru_14_gru_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_14/gru_cell_14/MatMul_1/ReadVariableOp?
gru_14/gru_cell_14/MatMul_1MatMulgru_14/zeros:output:02gru_14/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/MatMul_1?
gru_14/gru_cell_14/BiasAdd_1BiasAdd%gru_14/gru_cell_14/MatMul_1:product:0#gru_14/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/BiasAdd_1?
gru_14/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_14/gru_cell_14/Const?
$gru_14/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_14/gru_cell_14/split_1/split_dim?
gru_14/gru_cell_14/split_1SplitV%gru_14/gru_cell_14/BiasAdd_1:output:0!gru_14/gru_cell_14/Const:output:0-gru_14/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_14/gru_cell_14/split_1?
gru_14/gru_cell_14/addAddV2!gru_14/gru_cell_14/split:output:0#gru_14/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/add?
gru_14/gru_cell_14/SigmoidSigmoidgru_14/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/Sigmoid?
gru_14/gru_cell_14/add_1AddV2!gru_14/gru_cell_14/split:output:1#gru_14/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/add_1?
gru_14/gru_cell_14/Sigmoid_1Sigmoidgru_14/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/Sigmoid_1?
gru_14/gru_cell_14/mulMul gru_14/gru_cell_14/Sigmoid_1:y:0#gru_14/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/mul?
gru_14/gru_cell_14/add_2AddV2!gru_14/gru_cell_14/split:output:2gru_14/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/add_2?
gru_14/gru_cell_14/ReluRelugru_14/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/Relu?
gru_14/gru_cell_14/mul_1Mulgru_14/gru_cell_14/Sigmoid:y:0gru_14/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/mul_1y
gru_14/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_14/gru_cell_14/sub/x?
gru_14/gru_cell_14/subSub!gru_14/gru_cell_14/sub/x:output:0gru_14/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/sub?
gru_14/gru_cell_14/mul_2Mulgru_14/gru_cell_14/sub:z:0%gru_14/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/mul_2?
gru_14/gru_cell_14/add_3AddV2gru_14/gru_cell_14/mul_1:z:0gru_14/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/add_3?
$gru_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$gru_14/TensorArrayV2_1/element_shape?
gru_14/TensorArrayV2_1TensorListReserve-gru_14/TensorArrayV2_1/element_shape:output:0gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_14/TensorArrayV2_1\
gru_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_14/time?
gru_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_14/while/maximum_iterationsx
gru_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_14/while/loop_counter?
gru_14/whileWhile"gru_14/while/loop_counter:output:0(gru_14/while/maximum_iterations:output:0gru_14/time:output:0gru_14/TensorArrayV2_1:handle:0gru_14/zeros:output:0gru_14/strided_slice_1:output:0>gru_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_14_gru_cell_14_readvariableop_resource1gru_14_gru_cell_14_matmul_readvariableop_resource3gru_14_gru_cell_14_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *$
bodyR
gru_14_while_body_700116*$
condR
gru_14_while_cond_700115*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_14/while?
7gru_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7gru_14/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_14/TensorArrayV2Stack/TensorListStackTensorListStackgru_14/while:output:3@gru_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_14/TensorArrayV2Stack/TensorListStack?
gru_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_14/strided_slice_3/stack?
gru_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_14/strided_slice_3/stack_1?
gru_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_14/strided_slice_3/stack_2?
gru_14/strided_slice_3StridedSlice2gru_14/TensorArrayV2Stack/TensorListStack:tensor:0%gru_14/strided_slice_3/stack:output:0'gru_14/strided_slice_3/stack_1:output:0'gru_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_14/strided_slice_3?
gru_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_14/transpose_1/perm?
gru_14/transpose_1	Transpose2gru_14/TensorArrayV2Stack/TensorListStack:tensor:0 gru_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_14/transpose_1t
gru_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_14/runtimey
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_21/dropout/Const?
dropout_21/dropout/MulMulgru_14/transpose_1:y:0!dropout_21/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_21/dropout/Mulz
dropout_21/dropout/ShapeShapegru_14/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shape?
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_21/dropout/random_uniform/RandomUniform?
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_21/dropout/GreaterEqual/y?
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_21/dropout/GreaterEqual?
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_21/dropout/Cast?
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_21/dropout/Mul_1h
gru_15/ShapeShapedropout_21/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
gru_15/Shape?
gru_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_15/strided_slice/stack?
gru_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_15/strided_slice/stack_1?
gru_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_15/strided_slice/stack_2?
gru_15/strided_sliceStridedSlicegru_15/Shape:output:0#gru_15/strided_slice/stack:output:0%gru_15/strided_slice/stack_1:output:0%gru_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_15/strided_sliceq
gru_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_15/zeros/packed/1?
gru_15/zeros/packedPackgru_15/strided_slice:output:0gru_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_15/zeros/packedm
gru_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_15/zeros/Const?
gru_15/zerosFillgru_15/zeros/packed:output:0gru_15/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_15/zeros?
gru_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_15/transpose/perm?
gru_15/transpose	Transposedropout_21/dropout/Mul_1:z:0gru_15/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_15/transposed
gru_15/Shape_1Shapegru_15/transpose:y:0*
T0*
_output_shapes
:2
gru_15/Shape_1?
gru_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_15/strided_slice_1/stack?
gru_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_15/strided_slice_1/stack_1?
gru_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_15/strided_slice_1/stack_2?
gru_15/strided_slice_1StridedSlicegru_15/Shape_1:output:0%gru_15/strided_slice_1/stack:output:0'gru_15/strided_slice_1/stack_1:output:0'gru_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_15/strided_slice_1?
"gru_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_15/TensorArrayV2/element_shape?
gru_15/TensorArrayV2TensorListReserve+gru_15/TensorArrayV2/element_shape:output:0gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_15/TensorArrayV2?
<gru_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_15/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_15/transpose:y:0Egru_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_15/TensorArrayUnstack/TensorListFromTensor?
gru_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_15/strided_slice_2/stack?
gru_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_15/strided_slice_2/stack_1?
gru_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_15/strided_slice_2/stack_2?
gru_15/strided_slice_2StridedSlicegru_15/transpose:y:0%gru_15/strided_slice_2/stack:output:0'gru_15/strided_slice_2/stack_1:output:0'gru_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_15/strided_slice_2?
!gru_15/gru_cell_15/ReadVariableOpReadVariableOp*gru_15_gru_cell_15_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_15/gru_cell_15/ReadVariableOp?
gru_15/gru_cell_15/unstackUnpack)gru_15/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_15/gru_cell_15/unstack?
(gru_15/gru_cell_15/MatMul/ReadVariableOpReadVariableOp1gru_15_gru_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(gru_15/gru_cell_15/MatMul/ReadVariableOp?
gru_15/gru_cell_15/MatMulMatMulgru_15/strided_slice_2:output:00gru_15/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/MatMul?
gru_15/gru_cell_15/BiasAddBiasAdd#gru_15/gru_cell_15/MatMul:product:0#gru_15/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/BiasAdd?
"gru_15/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_15/gru_cell_15/split/split_dim?
gru_15/gru_cell_15/splitSplit+gru_15/gru_cell_15/split/split_dim:output:0#gru_15/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_15/gru_cell_15/split?
*gru_15/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp3gru_15_gru_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_15/gru_cell_15/MatMul_1/ReadVariableOp?
gru_15/gru_cell_15/MatMul_1MatMulgru_15/zeros:output:02gru_15/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/MatMul_1?
gru_15/gru_cell_15/BiasAdd_1BiasAdd%gru_15/gru_cell_15/MatMul_1:product:0#gru_15/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/BiasAdd_1?
gru_15/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_15/gru_cell_15/Const?
$gru_15/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_15/gru_cell_15/split_1/split_dim?
gru_15/gru_cell_15/split_1SplitV%gru_15/gru_cell_15/BiasAdd_1:output:0!gru_15/gru_cell_15/Const:output:0-gru_15/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_15/gru_cell_15/split_1?
gru_15/gru_cell_15/addAddV2!gru_15/gru_cell_15/split:output:0#gru_15/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/add?
gru_15/gru_cell_15/SigmoidSigmoidgru_15/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/Sigmoid?
gru_15/gru_cell_15/add_1AddV2!gru_15/gru_cell_15/split:output:1#gru_15/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/add_1?
gru_15/gru_cell_15/Sigmoid_1Sigmoidgru_15/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/Sigmoid_1?
gru_15/gru_cell_15/mulMul gru_15/gru_cell_15/Sigmoid_1:y:0#gru_15/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/mul?
gru_15/gru_cell_15/add_2AddV2!gru_15/gru_cell_15/split:output:2gru_15/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/add_2?
gru_15/gru_cell_15/ReluRelugru_15/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/Relu?
gru_15/gru_cell_15/mul_1Mulgru_15/gru_cell_15/Sigmoid:y:0gru_15/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/mul_1y
gru_15/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_15/gru_cell_15/sub/x?
gru_15/gru_cell_15/subSub!gru_15/gru_cell_15/sub/x:output:0gru_15/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/sub?
gru_15/gru_cell_15/mul_2Mulgru_15/gru_cell_15/sub:z:0%gru_15/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/mul_2?
gru_15/gru_cell_15/add_3AddV2gru_15/gru_cell_15/mul_1:z:0gru_15/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/add_3?
$gru_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$gru_15/TensorArrayV2_1/element_shape?
gru_15/TensorArrayV2_1TensorListReserve-gru_15/TensorArrayV2_1/element_shape:output:0gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_15/TensorArrayV2_1\
gru_15/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_15/time?
gru_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_15/while/maximum_iterationsx
gru_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_15/while/loop_counter?
gru_15/whileWhile"gru_15/while/loop_counter:output:0(gru_15/while/maximum_iterations:output:0gru_15/time:output:0gru_15/TensorArrayV2_1:handle:0gru_15/zeros:output:0gru_15/strided_slice_1:output:0>gru_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_15_gru_cell_15_readvariableop_resource1gru_15_gru_cell_15_matmul_readvariableop_resource3gru_15_gru_cell_15_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *$
bodyR
gru_15_while_body_700273*$
condR
gru_15_while_cond_700272*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_15/while?
7gru_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7gru_15/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_15/TensorArrayV2Stack/TensorListStackTensorListStackgru_15/while:output:3@gru_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_15/TensorArrayV2Stack/TensorListStack?
gru_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_15/strided_slice_3/stack?
gru_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_15/strided_slice_3/stack_1?
gru_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_15/strided_slice_3/stack_2?
gru_15/strided_slice_3StridedSlice2gru_15/TensorArrayV2Stack/TensorListStack:tensor:0%gru_15/strided_slice_3/stack:output:0'gru_15/strided_slice_3/stack_1:output:0'gru_15/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_15/strided_slice_3?
gru_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_15/transpose_1/perm?
gru_15/transpose_1	Transpose2gru_15/TensorArrayV2Stack/TensorListStack:tensor:0 gru_15/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_15/transpose_1t
gru_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_15/runtimey
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_22/dropout/Const?
dropout_22/dropout/MulMulgru_15/transpose_1:y:0!dropout_22/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_22/dropout/Mulz
dropout_22/dropout/ShapeShapegru_15/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shape?
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_22/dropout/random_uniform/RandomUniform?
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_22/dropout/GreaterEqual/y?
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_22/dropout/GreaterEqual?
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_22/dropout/Cast?
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_22/dropout/Mul_1?
!dense_14/Tensordot/ReadVariableOpReadVariableOp*dense_14_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_14/Tensordot/ReadVariableOp|
dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_14/Tensordot/axes?
dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_14/Tensordot/free?
dense_14/Tensordot/ShapeShapedropout_22/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_14/Tensordot/Shape?
 dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_14/Tensordot/GatherV2/axis?
dense_14/Tensordot/GatherV2GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/free:output:0)dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_14/Tensordot/GatherV2?
"dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_14/Tensordot/GatherV2_1/axis?
dense_14/Tensordot/GatherV2_1GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/axes:output:0+dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_14/Tensordot/GatherV2_1~
dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_14/Tensordot/Const?
dense_14/Tensordot/ProdProd$dense_14/Tensordot/GatherV2:output:0!dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_14/Tensordot/Prod?
dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_14/Tensordot/Const_1?
dense_14/Tensordot/Prod_1Prod&dense_14/Tensordot/GatherV2_1:output:0#dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_14/Tensordot/Prod_1?
dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_14/Tensordot/concat/axis?
dense_14/Tensordot/concatConcatV2 dense_14/Tensordot/free:output:0 dense_14/Tensordot/axes:output:0'dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_14/Tensordot/concat?
dense_14/Tensordot/stackPack dense_14/Tensordot/Prod:output:0"dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_14/Tensordot/stack?
dense_14/Tensordot/transpose	Transposedropout_22/dropout/Mul_1:z:0"dense_14/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_14/Tensordot/transpose?
dense_14/Tensordot/ReshapeReshape dense_14/Tensordot/transpose:y:0!dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_14/Tensordot/Reshape?
dense_14/Tensordot/MatMulMatMul#dense_14/Tensordot/Reshape:output:0)dense_14/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_14/Tensordot/MatMul?
dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_14/Tensordot/Const_2?
 dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_14/Tensordot/concat_1/axis?
dense_14/Tensordot/concat_1ConcatV2$dense_14/Tensordot/GatherV2:output:0#dense_14/Tensordot/Const_2:output:0)dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_14/Tensordot/concat_1?
dense_14/TensordotReshape#dense_14/Tensordot/MatMul:product:0$dense_14/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_14/Tensordot?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/Tensordot:output:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_14/BiasAddx
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_14/Reluy
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_23/dropout/Const?
dropout_23/dropout/MulMuldense_14/Relu:activations:0!dropout_23/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_23/dropout/Mul
dropout_23/dropout/ShapeShapedense_14/Relu:activations:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shape?
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_23/dropout/random_uniform/RandomUniform?
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_23/dropout/GreaterEqual/y?
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_23/dropout/GreaterEqual?
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_23/dropout/Cast?
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_23/dropout/Mul_1?
!dense_15/Tensordot/ReadVariableOpReadVariableOp*dense_15_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_15/Tensordot/ReadVariableOp|
dense_15/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_15/Tensordot/axes?
dense_15/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_15/Tensordot/free?
dense_15/Tensordot/ShapeShapedropout_23/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_15/Tensordot/Shape?
 dense_15/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_15/Tensordot/GatherV2/axis?
dense_15/Tensordot/GatherV2GatherV2!dense_15/Tensordot/Shape:output:0 dense_15/Tensordot/free:output:0)dense_15/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_15/Tensordot/GatherV2?
"dense_15/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_15/Tensordot/GatherV2_1/axis?
dense_15/Tensordot/GatherV2_1GatherV2!dense_15/Tensordot/Shape:output:0 dense_15/Tensordot/axes:output:0+dense_15/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_15/Tensordot/GatherV2_1~
dense_15/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_15/Tensordot/Const?
dense_15/Tensordot/ProdProd$dense_15/Tensordot/GatherV2:output:0!dense_15/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_15/Tensordot/Prod?
dense_15/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_15/Tensordot/Const_1?
dense_15/Tensordot/Prod_1Prod&dense_15/Tensordot/GatherV2_1:output:0#dense_15/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_15/Tensordot/Prod_1?
dense_15/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_15/Tensordot/concat/axis?
dense_15/Tensordot/concatConcatV2 dense_15/Tensordot/free:output:0 dense_15/Tensordot/axes:output:0'dense_15/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_15/Tensordot/concat?
dense_15/Tensordot/stackPack dense_15/Tensordot/Prod:output:0"dense_15/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_15/Tensordot/stack?
dense_15/Tensordot/transpose	Transposedropout_23/dropout/Mul_1:z:0"dense_15/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_15/Tensordot/transpose?
dense_15/Tensordot/ReshapeReshape dense_15/Tensordot/transpose:y:0!dense_15/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_15/Tensordot/Reshape?
dense_15/Tensordot/MatMulMatMul#dense_15/Tensordot/Reshape:output:0)dense_15/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_15/Tensordot/MatMul?
dense_15/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_15/Tensordot/Const_2?
 dense_15/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_15/Tensordot/concat_1/axis?
dense_15/Tensordot/concat_1ConcatV2$dense_15/Tensordot/GatherV2:output:0#dense_15/Tensordot/Const_2:output:0)dense_15/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_15/Tensordot/concat_1?
dense_15/TensordotReshape#dense_15/Tensordot/MatMul:product:0$dense_15/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_15/Tensordot?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/Tensordot:output:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_15/BiasAddx
IdentityIdentitydense_15/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp"^dense_14/Tensordot/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp"^dense_15/Tensordot/ReadVariableOp)^gru_14/gru_cell_14/MatMul/ReadVariableOp+^gru_14/gru_cell_14/MatMul_1/ReadVariableOp"^gru_14/gru_cell_14/ReadVariableOp^gru_14/while)^gru_15/gru_cell_15/MatMul/ReadVariableOp+^gru_15/gru_cell_15/MatMul_1/ReadVariableOp"^gru_15/gru_cell_15/ReadVariableOp^gru_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/Tensordot/ReadVariableOp!dense_14/Tensordot/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2F
!dense_15/Tensordot/ReadVariableOp!dense_15/Tensordot/ReadVariableOp2T
(gru_14/gru_cell_14/MatMul/ReadVariableOp(gru_14/gru_cell_14/MatMul/ReadVariableOp2X
*gru_14/gru_cell_14/MatMul_1/ReadVariableOp*gru_14/gru_cell_14/MatMul_1/ReadVariableOp2F
!gru_14/gru_cell_14/ReadVariableOp!gru_14/gru_cell_14/ReadVariableOp2
gru_14/whilegru_14/while2T
(gru_15/gru_cell_15/MatMul/ReadVariableOp(gru_15/gru_cell_15/MatMul/ReadVariableOp2X
*gru_15/gru_cell_15/MatMul_1/ReadVariableOp*gru_15/gru_cell_15/MatMul_1/ReadVariableOp2F
!gru_15/gru_cell_15/ReadVariableOp!gru_15/gru_cell_15/ReadVariableOp2
gru_15/whilegru_15/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
,__inference_gru_cell_15_layer_call_fn_702151

inputs
states_0
unknown:	?
	unknown_0:
??
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_6981132
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
while_cond_701533
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_701533___redundant_placeholder04
0while_while_cond_701533___redundant_placeholder14
0while_while_cond_701533___redundant_placeholder24
0while_while_cond_701533___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_697690

inputs

states*
readvariableop_resource:	?1
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
??
?
!__inference__wrapped_model_697477
gru_14_inputJ
7sequential_7_gru_14_gru_cell_14_readvariableop_resource:	?Q
>sequential_7_gru_14_gru_cell_14_matmul_readvariableop_resource:	?T
@sequential_7_gru_14_gru_cell_14_matmul_1_readvariableop_resource:
??J
7sequential_7_gru_15_gru_cell_15_readvariableop_resource:	?R
>sequential_7_gru_15_gru_cell_15_matmul_readvariableop_resource:
??T
@sequential_7_gru_15_gru_cell_15_matmul_1_readvariableop_resource:
??K
7sequential_7_dense_14_tensordot_readvariableop_resource:
??D
5sequential_7_dense_14_biasadd_readvariableop_resource:	?J
7sequential_7_dense_15_tensordot_readvariableop_resource:	?C
5sequential_7_dense_15_biasadd_readvariableop_resource:
identity??,sequential_7/dense_14/BiasAdd/ReadVariableOp?.sequential_7/dense_14/Tensordot/ReadVariableOp?,sequential_7/dense_15/BiasAdd/ReadVariableOp?.sequential_7/dense_15/Tensordot/ReadVariableOp?5sequential_7/gru_14/gru_cell_14/MatMul/ReadVariableOp?7sequential_7/gru_14/gru_cell_14/MatMul_1/ReadVariableOp?.sequential_7/gru_14/gru_cell_14/ReadVariableOp?sequential_7/gru_14/while?5sequential_7/gru_15/gru_cell_15/MatMul/ReadVariableOp?7sequential_7/gru_15/gru_cell_15/MatMul_1/ReadVariableOp?.sequential_7/gru_15/gru_cell_15/ReadVariableOp?sequential_7/gru_15/whiler
sequential_7/gru_14/ShapeShapegru_14_input*
T0*
_output_shapes
:2
sequential_7/gru_14/Shape?
'sequential_7/gru_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/gru_14/strided_slice/stack?
)sequential_7/gru_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_7/gru_14/strided_slice/stack_1?
)sequential_7/gru_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_7/gru_14/strided_slice/stack_2?
!sequential_7/gru_14/strided_sliceStridedSlice"sequential_7/gru_14/Shape:output:00sequential_7/gru_14/strided_slice/stack:output:02sequential_7/gru_14/strided_slice/stack_1:output:02sequential_7/gru_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_7/gru_14/strided_slice?
"sequential_7/gru_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_7/gru_14/zeros/packed/1?
 sequential_7/gru_14/zeros/packedPack*sequential_7/gru_14/strided_slice:output:0+sequential_7/gru_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_7/gru_14/zeros/packed?
sequential_7/gru_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_7/gru_14/zeros/Const?
sequential_7/gru_14/zerosFill)sequential_7/gru_14/zeros/packed:output:0(sequential_7/gru_14/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_7/gru_14/zeros?
"sequential_7/gru_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_7/gru_14/transpose/perm?
sequential_7/gru_14/transpose	Transposegru_14_input+sequential_7/gru_14/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
sequential_7/gru_14/transpose?
sequential_7/gru_14/Shape_1Shape!sequential_7/gru_14/transpose:y:0*
T0*
_output_shapes
:2
sequential_7/gru_14/Shape_1?
)sequential_7/gru_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_7/gru_14/strided_slice_1/stack?
+sequential_7/gru_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_7/gru_14/strided_slice_1/stack_1?
+sequential_7/gru_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_7/gru_14/strided_slice_1/stack_2?
#sequential_7/gru_14/strided_slice_1StridedSlice$sequential_7/gru_14/Shape_1:output:02sequential_7/gru_14/strided_slice_1/stack:output:04sequential_7/gru_14/strided_slice_1/stack_1:output:04sequential_7/gru_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_7/gru_14/strided_slice_1?
/sequential_7/gru_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_7/gru_14/TensorArrayV2/element_shape?
!sequential_7/gru_14/TensorArrayV2TensorListReserve8sequential_7/gru_14/TensorArrayV2/element_shape:output:0,sequential_7/gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_7/gru_14/TensorArrayV2?
Isequential_7/gru_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2K
Isequential_7/gru_14/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_7/gru_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_7/gru_14/transpose:y:0Rsequential_7/gru_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_7/gru_14/TensorArrayUnstack/TensorListFromTensor?
)sequential_7/gru_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_7/gru_14/strided_slice_2/stack?
+sequential_7/gru_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_7/gru_14/strided_slice_2/stack_1?
+sequential_7/gru_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_7/gru_14/strided_slice_2/stack_2?
#sequential_7/gru_14/strided_slice_2StridedSlice!sequential_7/gru_14/transpose:y:02sequential_7/gru_14/strided_slice_2/stack:output:04sequential_7/gru_14/strided_slice_2/stack_1:output:04sequential_7/gru_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2%
#sequential_7/gru_14/strided_slice_2?
.sequential_7/gru_14/gru_cell_14/ReadVariableOpReadVariableOp7sequential_7_gru_14_gru_cell_14_readvariableop_resource*
_output_shapes
:	?*
dtype020
.sequential_7/gru_14/gru_cell_14/ReadVariableOp?
'sequential_7/gru_14/gru_cell_14/unstackUnpack6sequential_7/gru_14/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2)
'sequential_7/gru_14/gru_cell_14/unstack?
5sequential_7/gru_14/gru_cell_14/MatMul/ReadVariableOpReadVariableOp>sequential_7_gru_14_gru_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype027
5sequential_7/gru_14/gru_cell_14/MatMul/ReadVariableOp?
&sequential_7/gru_14/gru_cell_14/MatMulMatMul,sequential_7/gru_14/strided_slice_2:output:0=sequential_7/gru_14/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential_7/gru_14/gru_cell_14/MatMul?
'sequential_7/gru_14/gru_cell_14/BiasAddBiasAdd0sequential_7/gru_14/gru_cell_14/MatMul:product:00sequential_7/gru_14/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2)
'sequential_7/gru_14/gru_cell_14/BiasAdd?
/sequential_7/gru_14/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_7/gru_14/gru_cell_14/split/split_dim?
%sequential_7/gru_14/gru_cell_14/splitSplit8sequential_7/gru_14/gru_cell_14/split/split_dim:output:00sequential_7/gru_14/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2'
%sequential_7/gru_14/gru_cell_14/split?
7sequential_7/gru_14/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp@sequential_7_gru_14_gru_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7sequential_7/gru_14/gru_cell_14/MatMul_1/ReadVariableOp?
(sequential_7/gru_14/gru_cell_14/MatMul_1MatMul"sequential_7/gru_14/zeros:output:0?sequential_7/gru_14/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_7/gru_14/gru_cell_14/MatMul_1?
)sequential_7/gru_14/gru_cell_14/BiasAdd_1BiasAdd2sequential_7/gru_14/gru_cell_14/MatMul_1:product:00sequential_7/gru_14/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2+
)sequential_7/gru_14/gru_cell_14/BiasAdd_1?
%sequential_7/gru_14/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2'
%sequential_7/gru_14/gru_cell_14/Const?
1sequential_7/gru_14/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_7/gru_14/gru_cell_14/split_1/split_dim?
'sequential_7/gru_14/gru_cell_14/split_1SplitV2sequential_7/gru_14/gru_cell_14/BiasAdd_1:output:0.sequential_7/gru_14/gru_cell_14/Const:output:0:sequential_7/gru_14/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2)
'sequential_7/gru_14/gru_cell_14/split_1?
#sequential_7/gru_14/gru_cell_14/addAddV2.sequential_7/gru_14/gru_cell_14/split:output:00sequential_7/gru_14/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2%
#sequential_7/gru_14/gru_cell_14/add?
'sequential_7/gru_14/gru_cell_14/SigmoidSigmoid'sequential_7/gru_14/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2)
'sequential_7/gru_14/gru_cell_14/Sigmoid?
%sequential_7/gru_14/gru_cell_14/add_1AddV2.sequential_7/gru_14/gru_cell_14/split:output:10sequential_7/gru_14/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2'
%sequential_7/gru_14/gru_cell_14/add_1?
)sequential_7/gru_14/gru_cell_14/Sigmoid_1Sigmoid)sequential_7/gru_14/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2+
)sequential_7/gru_14/gru_cell_14/Sigmoid_1?
#sequential_7/gru_14/gru_cell_14/mulMul-sequential_7/gru_14/gru_cell_14/Sigmoid_1:y:00sequential_7/gru_14/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2%
#sequential_7/gru_14/gru_cell_14/mul?
%sequential_7/gru_14/gru_cell_14/add_2AddV2.sequential_7/gru_14/gru_cell_14/split:output:2'sequential_7/gru_14/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2'
%sequential_7/gru_14/gru_cell_14/add_2?
$sequential_7/gru_14/gru_cell_14/ReluRelu)sequential_7/gru_14/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2&
$sequential_7/gru_14/gru_cell_14/Relu?
%sequential_7/gru_14/gru_cell_14/mul_1Mul+sequential_7/gru_14/gru_cell_14/Sigmoid:y:0"sequential_7/gru_14/zeros:output:0*
T0*(
_output_shapes
:??????????2'
%sequential_7/gru_14/gru_cell_14/mul_1?
%sequential_7/gru_14/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%sequential_7/gru_14/gru_cell_14/sub/x?
#sequential_7/gru_14/gru_cell_14/subSub.sequential_7/gru_14/gru_cell_14/sub/x:output:0+sequential_7/gru_14/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2%
#sequential_7/gru_14/gru_cell_14/sub?
%sequential_7/gru_14/gru_cell_14/mul_2Mul'sequential_7/gru_14/gru_cell_14/sub:z:02sequential_7/gru_14/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2'
%sequential_7/gru_14/gru_cell_14/mul_2?
%sequential_7/gru_14/gru_cell_14/add_3AddV2)sequential_7/gru_14/gru_cell_14/mul_1:z:0)sequential_7/gru_14/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2'
%sequential_7/gru_14/gru_cell_14/add_3?
1sequential_7/gru_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1sequential_7/gru_14/TensorArrayV2_1/element_shape?
#sequential_7/gru_14/TensorArrayV2_1TensorListReserve:sequential_7/gru_14/TensorArrayV2_1/element_shape:output:0,sequential_7/gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_7/gru_14/TensorArrayV2_1v
sequential_7/gru_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_7/gru_14/time?
,sequential_7/gru_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_7/gru_14/while/maximum_iterations?
&sequential_7/gru_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_7/gru_14/while/loop_counter?
sequential_7/gru_14/whileWhile/sequential_7/gru_14/while/loop_counter:output:05sequential_7/gru_14/while/maximum_iterations:output:0!sequential_7/gru_14/time:output:0,sequential_7/gru_14/TensorArrayV2_1:handle:0"sequential_7/gru_14/zeros:output:0,sequential_7/gru_14/strided_slice_1:output:0Ksequential_7/gru_14/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_7_gru_14_gru_cell_14_readvariableop_resource>sequential_7_gru_14_gru_cell_14_matmul_readvariableop_resource@sequential_7_gru_14_gru_cell_14_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *1
body)R'
%sequential_7_gru_14_while_body_697183*1
cond)R'
%sequential_7_gru_14_while_cond_697182*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
sequential_7/gru_14/while?
Dsequential_7/gru_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsequential_7/gru_14/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_7/gru_14/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_7/gru_14/while:output:3Msequential_7/gru_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype028
6sequential_7/gru_14/TensorArrayV2Stack/TensorListStack?
)sequential_7/gru_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_7/gru_14/strided_slice_3/stack?
+sequential_7/gru_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_7/gru_14/strided_slice_3/stack_1?
+sequential_7/gru_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_7/gru_14/strided_slice_3/stack_2?
#sequential_7/gru_14/strided_slice_3StridedSlice?sequential_7/gru_14/TensorArrayV2Stack/TensorListStack:tensor:02sequential_7/gru_14/strided_slice_3/stack:output:04sequential_7/gru_14/strided_slice_3/stack_1:output:04sequential_7/gru_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2%
#sequential_7/gru_14/strided_slice_3?
$sequential_7/gru_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_7/gru_14/transpose_1/perm?
sequential_7/gru_14/transpose_1	Transpose?sequential_7/gru_14/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_7/gru_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2!
sequential_7/gru_14/transpose_1?
sequential_7/gru_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_7/gru_14/runtime?
 sequential_7/dropout_21/IdentityIdentity#sequential_7/gru_14/transpose_1:y:0*
T0*,
_output_shapes
:??????????2"
 sequential_7/dropout_21/Identity?
sequential_7/gru_15/ShapeShape)sequential_7/dropout_21/Identity:output:0*
T0*
_output_shapes
:2
sequential_7/gru_15/Shape?
'sequential_7/gru_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/gru_15/strided_slice/stack?
)sequential_7/gru_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_7/gru_15/strided_slice/stack_1?
)sequential_7/gru_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_7/gru_15/strided_slice/stack_2?
!sequential_7/gru_15/strided_sliceStridedSlice"sequential_7/gru_15/Shape:output:00sequential_7/gru_15/strided_slice/stack:output:02sequential_7/gru_15/strided_slice/stack_1:output:02sequential_7/gru_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_7/gru_15/strided_slice?
"sequential_7/gru_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_7/gru_15/zeros/packed/1?
 sequential_7/gru_15/zeros/packedPack*sequential_7/gru_15/strided_slice:output:0+sequential_7/gru_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_7/gru_15/zeros/packed?
sequential_7/gru_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_7/gru_15/zeros/Const?
sequential_7/gru_15/zerosFill)sequential_7/gru_15/zeros/packed:output:0(sequential_7/gru_15/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_7/gru_15/zeros?
"sequential_7/gru_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_7/gru_15/transpose/perm?
sequential_7/gru_15/transpose	Transpose)sequential_7/dropout_21/Identity:output:0+sequential_7/gru_15/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
sequential_7/gru_15/transpose?
sequential_7/gru_15/Shape_1Shape!sequential_7/gru_15/transpose:y:0*
T0*
_output_shapes
:2
sequential_7/gru_15/Shape_1?
)sequential_7/gru_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_7/gru_15/strided_slice_1/stack?
+sequential_7/gru_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_7/gru_15/strided_slice_1/stack_1?
+sequential_7/gru_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_7/gru_15/strided_slice_1/stack_2?
#sequential_7/gru_15/strided_slice_1StridedSlice$sequential_7/gru_15/Shape_1:output:02sequential_7/gru_15/strided_slice_1/stack:output:04sequential_7/gru_15/strided_slice_1/stack_1:output:04sequential_7/gru_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_7/gru_15/strided_slice_1?
/sequential_7/gru_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_7/gru_15/TensorArrayV2/element_shape?
!sequential_7/gru_15/TensorArrayV2TensorListReserve8sequential_7/gru_15/TensorArrayV2/element_shape:output:0,sequential_7/gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_7/gru_15/TensorArrayV2?
Isequential_7/gru_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2K
Isequential_7/gru_15/TensorArrayUnstack/TensorListFromTensor/element_shape?
;sequential_7/gru_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_7/gru_15/transpose:y:0Rsequential_7/gru_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_7/gru_15/TensorArrayUnstack/TensorListFromTensor?
)sequential_7/gru_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_7/gru_15/strided_slice_2/stack?
+sequential_7/gru_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_7/gru_15/strided_slice_2/stack_1?
+sequential_7/gru_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_7/gru_15/strided_slice_2/stack_2?
#sequential_7/gru_15/strided_slice_2StridedSlice!sequential_7/gru_15/transpose:y:02sequential_7/gru_15/strided_slice_2/stack:output:04sequential_7/gru_15/strided_slice_2/stack_1:output:04sequential_7/gru_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2%
#sequential_7/gru_15/strided_slice_2?
.sequential_7/gru_15/gru_cell_15/ReadVariableOpReadVariableOp7sequential_7_gru_15_gru_cell_15_readvariableop_resource*
_output_shapes
:	?*
dtype020
.sequential_7/gru_15/gru_cell_15/ReadVariableOp?
'sequential_7/gru_15/gru_cell_15/unstackUnpack6sequential_7/gru_15/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2)
'sequential_7/gru_15/gru_cell_15/unstack?
5sequential_7/gru_15/gru_cell_15/MatMul/ReadVariableOpReadVariableOp>sequential_7_gru_15_gru_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype027
5sequential_7/gru_15/gru_cell_15/MatMul/ReadVariableOp?
&sequential_7/gru_15/gru_cell_15/MatMulMatMul,sequential_7/gru_15/strided_slice_2:output:0=sequential_7/gru_15/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential_7/gru_15/gru_cell_15/MatMul?
'sequential_7/gru_15/gru_cell_15/BiasAddBiasAdd0sequential_7/gru_15/gru_cell_15/MatMul:product:00sequential_7/gru_15/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2)
'sequential_7/gru_15/gru_cell_15/BiasAdd?
/sequential_7/gru_15/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/sequential_7/gru_15/gru_cell_15/split/split_dim?
%sequential_7/gru_15/gru_cell_15/splitSplit8sequential_7/gru_15/gru_cell_15/split/split_dim:output:00sequential_7/gru_15/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2'
%sequential_7/gru_15/gru_cell_15/split?
7sequential_7/gru_15/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp@sequential_7_gru_15_gru_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7sequential_7/gru_15/gru_cell_15/MatMul_1/ReadVariableOp?
(sequential_7/gru_15/gru_cell_15/MatMul_1MatMul"sequential_7/gru_15/zeros:output:0?sequential_7/gru_15/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_7/gru_15/gru_cell_15/MatMul_1?
)sequential_7/gru_15/gru_cell_15/BiasAdd_1BiasAdd2sequential_7/gru_15/gru_cell_15/MatMul_1:product:00sequential_7/gru_15/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2+
)sequential_7/gru_15/gru_cell_15/BiasAdd_1?
%sequential_7/gru_15/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2'
%sequential_7/gru_15/gru_cell_15/Const?
1sequential_7/gru_15/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential_7/gru_15/gru_cell_15/split_1/split_dim?
'sequential_7/gru_15/gru_cell_15/split_1SplitV2sequential_7/gru_15/gru_cell_15/BiasAdd_1:output:0.sequential_7/gru_15/gru_cell_15/Const:output:0:sequential_7/gru_15/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2)
'sequential_7/gru_15/gru_cell_15/split_1?
#sequential_7/gru_15/gru_cell_15/addAddV2.sequential_7/gru_15/gru_cell_15/split:output:00sequential_7/gru_15/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2%
#sequential_7/gru_15/gru_cell_15/add?
'sequential_7/gru_15/gru_cell_15/SigmoidSigmoid'sequential_7/gru_15/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2)
'sequential_7/gru_15/gru_cell_15/Sigmoid?
%sequential_7/gru_15/gru_cell_15/add_1AddV2.sequential_7/gru_15/gru_cell_15/split:output:10sequential_7/gru_15/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2'
%sequential_7/gru_15/gru_cell_15/add_1?
)sequential_7/gru_15/gru_cell_15/Sigmoid_1Sigmoid)sequential_7/gru_15/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2+
)sequential_7/gru_15/gru_cell_15/Sigmoid_1?
#sequential_7/gru_15/gru_cell_15/mulMul-sequential_7/gru_15/gru_cell_15/Sigmoid_1:y:00sequential_7/gru_15/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2%
#sequential_7/gru_15/gru_cell_15/mul?
%sequential_7/gru_15/gru_cell_15/add_2AddV2.sequential_7/gru_15/gru_cell_15/split:output:2'sequential_7/gru_15/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2'
%sequential_7/gru_15/gru_cell_15/add_2?
$sequential_7/gru_15/gru_cell_15/ReluRelu)sequential_7/gru_15/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2&
$sequential_7/gru_15/gru_cell_15/Relu?
%sequential_7/gru_15/gru_cell_15/mul_1Mul+sequential_7/gru_15/gru_cell_15/Sigmoid:y:0"sequential_7/gru_15/zeros:output:0*
T0*(
_output_shapes
:??????????2'
%sequential_7/gru_15/gru_cell_15/mul_1?
%sequential_7/gru_15/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%sequential_7/gru_15/gru_cell_15/sub/x?
#sequential_7/gru_15/gru_cell_15/subSub.sequential_7/gru_15/gru_cell_15/sub/x:output:0+sequential_7/gru_15/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2%
#sequential_7/gru_15/gru_cell_15/sub?
%sequential_7/gru_15/gru_cell_15/mul_2Mul'sequential_7/gru_15/gru_cell_15/sub:z:02sequential_7/gru_15/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2'
%sequential_7/gru_15/gru_cell_15/mul_2?
%sequential_7/gru_15/gru_cell_15/add_3AddV2)sequential_7/gru_15/gru_cell_15/mul_1:z:0)sequential_7/gru_15/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2'
%sequential_7/gru_15/gru_cell_15/add_3?
1sequential_7/gru_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1sequential_7/gru_15/TensorArrayV2_1/element_shape?
#sequential_7/gru_15/TensorArrayV2_1TensorListReserve:sequential_7/gru_15/TensorArrayV2_1/element_shape:output:0,sequential_7/gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_7/gru_15/TensorArrayV2_1v
sequential_7/gru_15/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_7/gru_15/time?
,sequential_7/gru_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_7/gru_15/while/maximum_iterations?
&sequential_7/gru_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_7/gru_15/while/loop_counter?
sequential_7/gru_15/whileWhile/sequential_7/gru_15/while/loop_counter:output:05sequential_7/gru_15/while/maximum_iterations:output:0!sequential_7/gru_15/time:output:0,sequential_7/gru_15/TensorArrayV2_1:handle:0"sequential_7/gru_15/zeros:output:0,sequential_7/gru_15/strided_slice_1:output:0Ksequential_7/gru_15/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_7_gru_15_gru_cell_15_readvariableop_resource>sequential_7_gru_15_gru_cell_15_matmul_readvariableop_resource@sequential_7_gru_15_gru_cell_15_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *1
body)R'
%sequential_7_gru_15_while_body_697333*1
cond)R'
%sequential_7_gru_15_while_cond_697332*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
sequential_7/gru_15/while?
Dsequential_7/gru_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsequential_7/gru_15/TensorArrayV2Stack/TensorListStack/element_shape?
6sequential_7/gru_15/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_7/gru_15/while:output:3Msequential_7/gru_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype028
6sequential_7/gru_15/TensorArrayV2Stack/TensorListStack?
)sequential_7/gru_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2+
)sequential_7/gru_15/strided_slice_3/stack?
+sequential_7/gru_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_7/gru_15/strided_slice_3/stack_1?
+sequential_7/gru_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_7/gru_15/strided_slice_3/stack_2?
#sequential_7/gru_15/strided_slice_3StridedSlice?sequential_7/gru_15/TensorArrayV2Stack/TensorListStack:tensor:02sequential_7/gru_15/strided_slice_3/stack:output:04sequential_7/gru_15/strided_slice_3/stack_1:output:04sequential_7/gru_15/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2%
#sequential_7/gru_15/strided_slice_3?
$sequential_7/gru_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_7/gru_15/transpose_1/perm?
sequential_7/gru_15/transpose_1	Transpose?sequential_7/gru_15/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_7/gru_15/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2!
sequential_7/gru_15/transpose_1?
sequential_7/gru_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_7/gru_15/runtime?
 sequential_7/dropout_22/IdentityIdentity#sequential_7/gru_15/transpose_1:y:0*
T0*,
_output_shapes
:??????????2"
 sequential_7/dropout_22/Identity?
.sequential_7/dense_14/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_14_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.sequential_7/dense_14/Tensordot/ReadVariableOp?
$sequential_7/dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_14/Tensordot/axes?
$sequential_7/dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_14/Tensordot/free?
%sequential_7/dense_14/Tensordot/ShapeShape)sequential_7/dropout_22/Identity:output:0*
T0*
_output_shapes
:2'
%sequential_7/dense_14/Tensordot/Shape?
-sequential_7/dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_14/Tensordot/GatherV2/axis?
(sequential_7/dense_14/Tensordot/GatherV2GatherV2.sequential_7/dense_14/Tensordot/Shape:output:0-sequential_7/dense_14/Tensordot/free:output:06sequential_7/dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_14/Tensordot/GatherV2?
/sequential_7/dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_14/Tensordot/GatherV2_1/axis?
*sequential_7/dense_14/Tensordot/GatherV2_1GatherV2.sequential_7/dense_14/Tensordot/Shape:output:0-sequential_7/dense_14/Tensordot/axes:output:08sequential_7/dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_14/Tensordot/GatherV2_1?
%sequential_7/dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_14/Tensordot/Const?
$sequential_7/dense_14/Tensordot/ProdProd1sequential_7/dense_14/Tensordot/GatherV2:output:0.sequential_7/dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_14/Tensordot/Prod?
'sequential_7/dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_14/Tensordot/Const_1?
&sequential_7/dense_14/Tensordot/Prod_1Prod3sequential_7/dense_14/Tensordot/GatherV2_1:output:00sequential_7/dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_14/Tensordot/Prod_1?
+sequential_7/dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_14/Tensordot/concat/axis?
&sequential_7/dense_14/Tensordot/concatConcatV2-sequential_7/dense_14/Tensordot/free:output:0-sequential_7/dense_14/Tensordot/axes:output:04sequential_7/dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_14/Tensordot/concat?
%sequential_7/dense_14/Tensordot/stackPack-sequential_7/dense_14/Tensordot/Prod:output:0/sequential_7/dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_14/Tensordot/stack?
)sequential_7/dense_14/Tensordot/transpose	Transpose)sequential_7/dropout_22/Identity:output:0/sequential_7/dense_14/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)sequential_7/dense_14/Tensordot/transpose?
'sequential_7/dense_14/Tensordot/ReshapeReshape-sequential_7/dense_14/Tensordot/transpose:y:0.sequential_7/dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_7/dense_14/Tensordot/Reshape?
&sequential_7/dense_14/Tensordot/MatMulMatMul0sequential_7/dense_14/Tensordot/Reshape:output:06sequential_7/dense_14/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&sequential_7/dense_14/Tensordot/MatMul?
'sequential_7/dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2)
'sequential_7/dense_14/Tensordot/Const_2?
-sequential_7/dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_14/Tensordot/concat_1/axis?
(sequential_7/dense_14/Tensordot/concat_1ConcatV21sequential_7/dense_14/Tensordot/GatherV2:output:00sequential_7/dense_14/Tensordot/Const_2:output:06sequential_7/dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_14/Tensordot/concat_1?
sequential_7/dense_14/TensordotReshape0sequential_7/dense_14/Tensordot/MatMul:product:01sequential_7/dense_14/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2!
sequential_7/dense_14/Tensordot?
,sequential_7/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_7/dense_14/BiasAdd/ReadVariableOp?
sequential_7/dense_14/BiasAddBiasAdd(sequential_7/dense_14/Tensordot:output:04sequential_7/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
sequential_7/dense_14/BiasAdd?
sequential_7/dense_14/ReluRelu&sequential_7/dense_14/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
sequential_7/dense_14/Relu?
 sequential_7/dropout_23/IdentityIdentity(sequential_7/dense_14/Relu:activations:0*
T0*,
_output_shapes
:??????????2"
 sequential_7/dropout_23/Identity?
.sequential_7/dense_15/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_15_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype020
.sequential_7/dense_15/Tensordot/ReadVariableOp?
$sequential_7/dense_15/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_15/Tensordot/axes?
$sequential_7/dense_15/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_15/Tensordot/free?
%sequential_7/dense_15/Tensordot/ShapeShape)sequential_7/dropout_23/Identity:output:0*
T0*
_output_shapes
:2'
%sequential_7/dense_15/Tensordot/Shape?
-sequential_7/dense_15/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_15/Tensordot/GatherV2/axis?
(sequential_7/dense_15/Tensordot/GatherV2GatherV2.sequential_7/dense_15/Tensordot/Shape:output:0-sequential_7/dense_15/Tensordot/free:output:06sequential_7/dense_15/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_15/Tensordot/GatherV2?
/sequential_7/dense_15/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_15/Tensordot/GatherV2_1/axis?
*sequential_7/dense_15/Tensordot/GatherV2_1GatherV2.sequential_7/dense_15/Tensordot/Shape:output:0-sequential_7/dense_15/Tensordot/axes:output:08sequential_7/dense_15/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_15/Tensordot/GatherV2_1?
%sequential_7/dense_15/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_15/Tensordot/Const?
$sequential_7/dense_15/Tensordot/ProdProd1sequential_7/dense_15/Tensordot/GatherV2:output:0.sequential_7/dense_15/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_15/Tensordot/Prod?
'sequential_7/dense_15/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_15/Tensordot/Const_1?
&sequential_7/dense_15/Tensordot/Prod_1Prod3sequential_7/dense_15/Tensordot/GatherV2_1:output:00sequential_7/dense_15/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_15/Tensordot/Prod_1?
+sequential_7/dense_15/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_15/Tensordot/concat/axis?
&sequential_7/dense_15/Tensordot/concatConcatV2-sequential_7/dense_15/Tensordot/free:output:0-sequential_7/dense_15/Tensordot/axes:output:04sequential_7/dense_15/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_15/Tensordot/concat?
%sequential_7/dense_15/Tensordot/stackPack-sequential_7/dense_15/Tensordot/Prod:output:0/sequential_7/dense_15/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_15/Tensordot/stack?
)sequential_7/dense_15/Tensordot/transpose	Transpose)sequential_7/dropout_23/Identity:output:0/sequential_7/dense_15/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2+
)sequential_7/dense_15/Tensordot/transpose?
'sequential_7/dense_15/Tensordot/ReshapeReshape-sequential_7/dense_15/Tensordot/transpose:y:0.sequential_7/dense_15/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_7/dense_15/Tensordot/Reshape?
&sequential_7/dense_15/Tensordot/MatMulMatMul0sequential_7/dense_15/Tensordot/Reshape:output:06sequential_7/dense_15/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&sequential_7/dense_15/Tensordot/MatMul?
'sequential_7/dense_15/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential_7/dense_15/Tensordot/Const_2?
-sequential_7/dense_15/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_15/Tensordot/concat_1/axis?
(sequential_7/dense_15/Tensordot/concat_1ConcatV21sequential_7/dense_15/Tensordot/GatherV2:output:00sequential_7/dense_15/Tensordot/Const_2:output:06sequential_7/dense_15/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_15/Tensordot/concat_1?
sequential_7/dense_15/TensordotReshape0sequential_7/dense_15/Tensordot/MatMul:product:01sequential_7/dense_15/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2!
sequential_7/dense_15/Tensordot?
,sequential_7/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_7/dense_15/BiasAdd/ReadVariableOp?
sequential_7/dense_15/BiasAddBiasAdd(sequential_7/dense_15/Tensordot:output:04sequential_7/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_7/dense_15/BiasAdd?
IdentityIdentity&sequential_7/dense_15/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp-^sequential_7/dense_14/BiasAdd/ReadVariableOp/^sequential_7/dense_14/Tensordot/ReadVariableOp-^sequential_7/dense_15/BiasAdd/ReadVariableOp/^sequential_7/dense_15/Tensordot/ReadVariableOp6^sequential_7/gru_14/gru_cell_14/MatMul/ReadVariableOp8^sequential_7/gru_14/gru_cell_14/MatMul_1/ReadVariableOp/^sequential_7/gru_14/gru_cell_14/ReadVariableOp^sequential_7/gru_14/while6^sequential_7/gru_15/gru_cell_15/MatMul/ReadVariableOp8^sequential_7/gru_15/gru_cell_15/MatMul_1/ReadVariableOp/^sequential_7/gru_15/gru_cell_15/ReadVariableOp^sequential_7/gru_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2\
,sequential_7/dense_14/BiasAdd/ReadVariableOp,sequential_7/dense_14/BiasAdd/ReadVariableOp2`
.sequential_7/dense_14/Tensordot/ReadVariableOp.sequential_7/dense_14/Tensordot/ReadVariableOp2\
,sequential_7/dense_15/BiasAdd/ReadVariableOp,sequential_7/dense_15/BiasAdd/ReadVariableOp2`
.sequential_7/dense_15/Tensordot/ReadVariableOp.sequential_7/dense_15/Tensordot/ReadVariableOp2n
5sequential_7/gru_14/gru_cell_14/MatMul/ReadVariableOp5sequential_7/gru_14/gru_cell_14/MatMul/ReadVariableOp2r
7sequential_7/gru_14/gru_cell_14/MatMul_1/ReadVariableOp7sequential_7/gru_14/gru_cell_14/MatMul_1/ReadVariableOp2`
.sequential_7/gru_14/gru_cell_14/ReadVariableOp.sequential_7/gru_14/gru_cell_14/ReadVariableOp26
sequential_7/gru_14/whilesequential_7/gru_14/while2n
5sequential_7/gru_15/gru_cell_15/MatMul/ReadVariableOp5sequential_7/gru_15/gru_cell_15/MatMul/ReadVariableOp2r
7sequential_7/gru_15/gru_cell_15/MatMul_1/ReadVariableOp7sequential_7/gru_15/gru_cell_15/MatMul_1/ReadVariableOp2`
.sequential_7/gru_15/gru_cell_15/ReadVariableOp.sequential_7/gru_15/gru_cell_15/ReadVariableOp26
sequential_7/gru_15/whilesequential_7/gru_15/while:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_14_input
?;
?
B__inference_gru_15_layer_call_and_return_conditional_losses_698383

inputs%
gru_cell_15_698307:	?&
gru_cell_15_698309:
??&
gru_cell_15_698311:
??
identity??#gru_cell_15/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_15/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_15_698307gru_cell_15_698309gru_cell_15_698311*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_6982562%
#gru_cell_15/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_15_698307gru_cell_15_698309gru_cell_15_698311*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_698319*
condR
while_cond_698318*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity|
NoOpNoOp$^gru_cell_15/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#gru_cell_15/StatefulPartitionedCall#gru_cell_15/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_701992

inputs
states_0*
readvariableop_resource:	?1
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?d
?
%sequential_7_gru_14_while_body_697183D
@sequential_7_gru_14_while_sequential_7_gru_14_while_loop_counterJ
Fsequential_7_gru_14_while_sequential_7_gru_14_while_maximum_iterations)
%sequential_7_gru_14_while_placeholder+
'sequential_7_gru_14_while_placeholder_1+
'sequential_7_gru_14_while_placeholder_2C
?sequential_7_gru_14_while_sequential_7_gru_14_strided_slice_1_0
{sequential_7_gru_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_gru_14_tensorarrayunstack_tensorlistfromtensor_0R
?sequential_7_gru_14_while_gru_cell_14_readvariableop_resource_0:	?Y
Fsequential_7_gru_14_while_gru_cell_14_matmul_readvariableop_resource_0:	?\
Hsequential_7_gru_14_while_gru_cell_14_matmul_1_readvariableop_resource_0:
??&
"sequential_7_gru_14_while_identity(
$sequential_7_gru_14_while_identity_1(
$sequential_7_gru_14_while_identity_2(
$sequential_7_gru_14_while_identity_3(
$sequential_7_gru_14_while_identity_4A
=sequential_7_gru_14_while_sequential_7_gru_14_strided_slice_1}
ysequential_7_gru_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_gru_14_tensorarrayunstack_tensorlistfromtensorP
=sequential_7_gru_14_while_gru_cell_14_readvariableop_resource:	?W
Dsequential_7_gru_14_while_gru_cell_14_matmul_readvariableop_resource:	?Z
Fsequential_7_gru_14_while_gru_cell_14_matmul_1_readvariableop_resource:
????;sequential_7/gru_14/while/gru_cell_14/MatMul/ReadVariableOp?=sequential_7/gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp?4sequential_7/gru_14/while/gru_cell_14/ReadVariableOp?
Ksequential_7/gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2M
Ksequential_7/gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shape?
=sequential_7/gru_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_7_gru_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_gru_14_tensorarrayunstack_tensorlistfromtensor_0%sequential_7_gru_14_while_placeholderTsequential_7/gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02?
=sequential_7/gru_14/while/TensorArrayV2Read/TensorListGetItem?
4sequential_7/gru_14/while/gru_cell_14/ReadVariableOpReadVariableOp?sequential_7_gru_14_while_gru_cell_14_readvariableop_resource_0*
_output_shapes
:	?*
dtype026
4sequential_7/gru_14/while/gru_cell_14/ReadVariableOp?
-sequential_7/gru_14/while/gru_cell_14/unstackUnpack<sequential_7/gru_14/while/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2/
-sequential_7/gru_14/while/gru_cell_14/unstack?
;sequential_7/gru_14/while/gru_cell_14/MatMul/ReadVariableOpReadVariableOpFsequential_7_gru_14_while_gru_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02=
;sequential_7/gru_14/while/gru_cell_14/MatMul/ReadVariableOp?
,sequential_7/gru_14/while/gru_cell_14/MatMulMatMulDsequential_7/gru_14/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_7/gru_14/while/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,sequential_7/gru_14/while/gru_cell_14/MatMul?
-sequential_7/gru_14/while/gru_cell_14/BiasAddBiasAdd6sequential_7/gru_14/while/gru_cell_14/MatMul:product:06sequential_7/gru_14/while/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2/
-sequential_7/gru_14/while/gru_cell_14/BiasAdd?
5sequential_7/gru_14/while/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????27
5sequential_7/gru_14/while/gru_cell_14/split/split_dim?
+sequential_7/gru_14/while/gru_cell_14/splitSplit>sequential_7/gru_14/while/gru_cell_14/split/split_dim:output:06sequential_7/gru_14/while/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2-
+sequential_7/gru_14/while/gru_cell_14/split?
=sequential_7/gru_14/while/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOpHsequential_7_gru_14_while_gru_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02?
=sequential_7/gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp?
.sequential_7/gru_14/while/gru_cell_14/MatMul_1MatMul'sequential_7_gru_14_while_placeholder_2Esequential_7/gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_7/gru_14/while/gru_cell_14/MatMul_1?
/sequential_7/gru_14/while/gru_cell_14/BiasAdd_1BiasAdd8sequential_7/gru_14/while/gru_cell_14/MatMul_1:product:06sequential_7/gru_14/while/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????21
/sequential_7/gru_14/while/gru_cell_14/BiasAdd_1?
+sequential_7/gru_14/while/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2-
+sequential_7/gru_14/while/gru_cell_14/Const?
7sequential_7/gru_14/while/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????29
7sequential_7/gru_14/while/gru_cell_14/split_1/split_dim?
-sequential_7/gru_14/while/gru_cell_14/split_1SplitV8sequential_7/gru_14/while/gru_cell_14/BiasAdd_1:output:04sequential_7/gru_14/while/gru_cell_14/Const:output:0@sequential_7/gru_14/while/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2/
-sequential_7/gru_14/while/gru_cell_14/split_1?
)sequential_7/gru_14/while/gru_cell_14/addAddV24sequential_7/gru_14/while/gru_cell_14/split:output:06sequential_7/gru_14/while/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_7/gru_14/while/gru_cell_14/add?
-sequential_7/gru_14/while/gru_cell_14/SigmoidSigmoid-sequential_7/gru_14/while/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2/
-sequential_7/gru_14/while/gru_cell_14/Sigmoid?
+sequential_7/gru_14/while/gru_cell_14/add_1AddV24sequential_7/gru_14/while/gru_cell_14/split:output:16sequential_7/gru_14/while/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2-
+sequential_7/gru_14/while/gru_cell_14/add_1?
/sequential_7/gru_14/while/gru_cell_14/Sigmoid_1Sigmoid/sequential_7/gru_14/while/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????21
/sequential_7/gru_14/while/gru_cell_14/Sigmoid_1?
)sequential_7/gru_14/while/gru_cell_14/mulMul3sequential_7/gru_14/while/gru_cell_14/Sigmoid_1:y:06sequential_7/gru_14/while/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2+
)sequential_7/gru_14/while/gru_cell_14/mul?
+sequential_7/gru_14/while/gru_cell_14/add_2AddV24sequential_7/gru_14/while/gru_cell_14/split:output:2-sequential_7/gru_14/while/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2-
+sequential_7/gru_14/while/gru_cell_14/add_2?
*sequential_7/gru_14/while/gru_cell_14/ReluRelu/sequential_7/gru_14/while/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2,
*sequential_7/gru_14/while/gru_cell_14/Relu?
+sequential_7/gru_14/while/gru_cell_14/mul_1Mul1sequential_7/gru_14/while/gru_cell_14/Sigmoid:y:0'sequential_7_gru_14_while_placeholder_2*
T0*(
_output_shapes
:??????????2-
+sequential_7/gru_14/while/gru_cell_14/mul_1?
+sequential_7/gru_14/while/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+sequential_7/gru_14/while/gru_cell_14/sub/x?
)sequential_7/gru_14/while/gru_cell_14/subSub4sequential_7/gru_14/while/gru_cell_14/sub/x:output:01sequential_7/gru_14/while/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2+
)sequential_7/gru_14/while/gru_cell_14/sub?
+sequential_7/gru_14/while/gru_cell_14/mul_2Mul-sequential_7/gru_14/while/gru_cell_14/sub:z:08sequential_7/gru_14/while/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2-
+sequential_7/gru_14/while/gru_cell_14/mul_2?
+sequential_7/gru_14/while/gru_cell_14/add_3AddV2/sequential_7/gru_14/while/gru_cell_14/mul_1:z:0/sequential_7/gru_14/while/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2-
+sequential_7/gru_14/while/gru_cell_14/add_3?
>sequential_7/gru_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_7_gru_14_while_placeholder_1%sequential_7_gru_14_while_placeholder/sequential_7/gru_14/while/gru_cell_14/add_3:z:0*
_output_shapes
: *
element_dtype02@
>sequential_7/gru_14/while/TensorArrayV2Write/TensorListSetItem?
sequential_7/gru_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_7/gru_14/while/add/y?
sequential_7/gru_14/while/addAddV2%sequential_7_gru_14_while_placeholder(sequential_7/gru_14/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_7/gru_14/while/add?
!sequential_7/gru_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_7/gru_14/while/add_1/y?
sequential_7/gru_14/while/add_1AddV2@sequential_7_gru_14_while_sequential_7_gru_14_while_loop_counter*sequential_7/gru_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_7/gru_14/while/add_1?
"sequential_7/gru_14/while/IdentityIdentity#sequential_7/gru_14/while/add_1:z:0^sequential_7/gru_14/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_7/gru_14/while/Identity?
$sequential_7/gru_14/while/Identity_1IdentityFsequential_7_gru_14_while_sequential_7_gru_14_while_maximum_iterations^sequential_7/gru_14/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_7/gru_14/while/Identity_1?
$sequential_7/gru_14/while/Identity_2Identity!sequential_7/gru_14/while/add:z:0^sequential_7/gru_14/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_7/gru_14/while/Identity_2?
$sequential_7/gru_14/while/Identity_3IdentityNsequential_7/gru_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_7/gru_14/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_7/gru_14/while/Identity_3?
$sequential_7/gru_14/while/Identity_4Identity/sequential_7/gru_14/while/gru_cell_14/add_3:z:0^sequential_7/gru_14/while/NoOp*
T0*(
_output_shapes
:??????????2&
$sequential_7/gru_14/while/Identity_4?
sequential_7/gru_14/while/NoOpNoOp<^sequential_7/gru_14/while/gru_cell_14/MatMul/ReadVariableOp>^sequential_7/gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp5^sequential_7/gru_14/while/gru_cell_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_7/gru_14/while/NoOp"?
Fsequential_7_gru_14_while_gru_cell_14_matmul_1_readvariableop_resourceHsequential_7_gru_14_while_gru_cell_14_matmul_1_readvariableop_resource_0"?
Dsequential_7_gru_14_while_gru_cell_14_matmul_readvariableop_resourceFsequential_7_gru_14_while_gru_cell_14_matmul_readvariableop_resource_0"?
=sequential_7_gru_14_while_gru_cell_14_readvariableop_resource?sequential_7_gru_14_while_gru_cell_14_readvariableop_resource_0"Q
"sequential_7_gru_14_while_identity+sequential_7/gru_14/while/Identity:output:0"U
$sequential_7_gru_14_while_identity_1-sequential_7/gru_14/while/Identity_1:output:0"U
$sequential_7_gru_14_while_identity_2-sequential_7/gru_14/while/Identity_2:output:0"U
$sequential_7_gru_14_while_identity_3-sequential_7/gru_14/while/Identity_3:output:0"U
$sequential_7_gru_14_while_identity_4-sequential_7/gru_14/while/Identity_4:output:0"?
=sequential_7_gru_14_while_sequential_7_gru_14_strided_slice_1?sequential_7_gru_14_while_sequential_7_gru_14_strided_slice_1_0"?
ysequential_7_gru_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_gru_14_tensorarrayunstack_tensorlistfromtensor{sequential_7_gru_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_gru_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2z
;sequential_7/gru_14/while/gru_cell_14/MatMul/ReadVariableOp;sequential_7/gru_14/while/gru_cell_14/MatMul/ReadVariableOp2~
=sequential_7/gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp=sequential_7/gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp2l
4sequential_7/gru_14/while/gru_cell_14/ReadVariableOp4sequential_7/gru_14/while/gru_cell_14/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_700544
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_700544___redundant_placeholder04
0while_while_cond_700544___redundant_placeholder14
0while_while_cond_700544___redundant_placeholder24
0while_while_cond_700544___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
G
+__inference_dropout_22_layer_call_fn_701842

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6989482
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_dense_15_layer_call_fn_701953

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_6990242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
-__inference_sequential_7_layer_call_fn_700481

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_6995512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?Y
?
B__inference_gru_14_layer_call_and_return_conditional_losses_700787
inputs_06
#gru_cell_14_readvariableop_resource:	?=
*gru_cell_14_matmul_readvariableop_resource:	?@
,gru_cell_14_matmul_1_readvariableop_resource:
??
identity??!gru_cell_14/MatMul/ReadVariableOp?#gru_cell_14/MatMul_1/ReadVariableOp?gru_cell_14/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_14/ReadVariableOpReadVariableOp#gru_cell_14_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_14/ReadVariableOp?
gru_cell_14/unstackUnpack"gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_14/unstack?
!gru_cell_14/MatMul/ReadVariableOpReadVariableOp*gru_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_14/MatMul/ReadVariableOp?
gru_cell_14/MatMulMatMulstrided_slice_2:output:0)gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul?
gru_cell_14/BiasAddBiasAddgru_cell_14/MatMul:product:0gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd?
gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split/split_dim?
gru_cell_14/splitSplit$gru_cell_14/split/split_dim:output:0gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split?
#gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_14/MatMul_1/ReadVariableOp?
gru_cell_14/MatMul_1MatMulzeros:output:0+gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul_1?
gru_cell_14/BiasAdd_1BiasAddgru_cell_14/MatMul_1:product:0gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd_1{
gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_14/Const?
gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split_1/split_dim?
gru_cell_14/split_1SplitVgru_cell_14/BiasAdd_1:output:0gru_cell_14/Const:output:0&gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split_1?
gru_cell_14/addAddV2gru_cell_14/split:output:0gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add}
gru_cell_14/SigmoidSigmoidgru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid?
gru_cell_14/add_1AddV2gru_cell_14/split:output:1gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_1?
gru_cell_14/Sigmoid_1Sigmoidgru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid_1?
gru_cell_14/mulMulgru_cell_14/Sigmoid_1:y:0gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul?
gru_cell_14/add_2AddV2gru_cell_14/split:output:2gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_2v
gru_cell_14/ReluRelugru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Relu?
gru_cell_14/mul_1Mulgru_cell_14/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_1k
gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_14/sub/x?
gru_cell_14/subSubgru_cell_14/sub/x:output:0gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/sub?
gru_cell_14/mul_2Mulgru_cell_14/sub:z:0gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_2?
gru_cell_14/add_3AddV2gru_cell_14/mul_1:z:0gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_14_readvariableop_resource*gru_cell_14_matmul_readvariableop_resource,gru_cell_14_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_700698*
condR
while_cond_700697*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp"^gru_cell_14/MatMul/ReadVariableOp$^gru_cell_14/MatMul_1/ReadVariableOp^gru_cell_14/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2F
!gru_cell_14/MatMul/ReadVariableOp!gru_cell_14/MatMul/ReadVariableOp2J
#gru_cell_14/MatMul_1/ReadVariableOp#gru_cell_14/MatMul_1/ReadVariableOp28
gru_cell_14/ReadVariableOpgru_cell_14/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?;
?
B__inference_gru_15_layer_call_and_return_conditional_losses_698190

inputs%
gru_cell_15_698114:	?&
gru_cell_15_698116:
??&
gru_cell_15_698118:
??
identity??#gru_cell_15/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_15/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_15_698114gru_cell_15_698116gru_cell_15_698118*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_6981132%
#gru_cell_15/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_15_698114gru_cell_15_698116gru_cell_15_698118*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_698126*
condR
while_cond_698125*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity|
NoOpNoOp$^gru_cell_15/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#gru_cell_15/StatefulPartitionedCall#gru_cell_15/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
˫
?
"__inference__traced_restore_702432
file_prefix4
 assignvariableop_dense_14_kernel:
??/
 assignvariableop_1_dense_14_bias:	?5
"assignvariableop_2_dense_15_kernel:	?.
 assignvariableop_3_dense_15_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: ?
,assignvariableop_9_gru_14_gru_cell_14_kernel:	?K
7assignvariableop_10_gru_14_gru_cell_14_recurrent_kernel:
??>
+assignvariableop_11_gru_14_gru_cell_14_bias:	?A
-assignvariableop_12_gru_15_gru_cell_15_kernel:
??K
7assignvariableop_13_gru_15_gru_cell_15_recurrent_kernel:
??>
+assignvariableop_14_gru_15_gru_cell_15_bias:	?#
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: >
*assignvariableop_19_adam_dense_14_kernel_m:
??7
(assignvariableop_20_adam_dense_14_bias_m:	?=
*assignvariableop_21_adam_dense_15_kernel_m:	?6
(assignvariableop_22_adam_dense_15_bias_m:G
4assignvariableop_23_adam_gru_14_gru_cell_14_kernel_m:	?R
>assignvariableop_24_adam_gru_14_gru_cell_14_recurrent_kernel_m:
??E
2assignvariableop_25_adam_gru_14_gru_cell_14_bias_m:	?H
4assignvariableop_26_adam_gru_15_gru_cell_15_kernel_m:
??R
>assignvariableop_27_adam_gru_15_gru_cell_15_recurrent_kernel_m:
??E
2assignvariableop_28_adam_gru_15_gru_cell_15_bias_m:	?>
*assignvariableop_29_adam_dense_14_kernel_v:
??7
(assignvariableop_30_adam_dense_14_bias_v:	?=
*assignvariableop_31_adam_dense_15_kernel_v:	?6
(assignvariableop_32_adam_dense_15_bias_v:G
4assignvariableop_33_adam_gru_14_gru_cell_14_kernel_v:	?R
>assignvariableop_34_adam_gru_14_gru_cell_14_recurrent_kernel_v:
??E
2assignvariableop_35_adam_gru_14_gru_cell_14_bias_v:	?H
4assignvariableop_36_adam_gru_15_gru_cell_15_kernel_v:
??R
>assignvariableop_37_adam_gru_15_gru_cell_15_recurrent_kernel_v:
??E
2assignvariableop_38_adam_gru_15_gru_cell_15_bias_v:	?
identity_40??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_15_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_15_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_gru_14_gru_cell_14_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_gru_14_gru_cell_14_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_gru_14_gru_cell_14_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp-assignvariableop_12_gru_15_gru_cell_15_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp7assignvariableop_13_gru_15_gru_cell_15_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp+assignvariableop_14_gru_15_gru_cell_15_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_14_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_14_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_15_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_15_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_gru_14_gru_cell_14_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp>assignvariableop_24_adam_gru_14_gru_cell_14_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_gru_14_gru_cell_14_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_gru_15_gru_cell_15_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_gru_15_gru_cell_15_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp2assignvariableop_28_adam_gru_15_gru_cell_15_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_14_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_14_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_15_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_15_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_adam_gru_14_gru_cell_14_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp>assignvariableop_34_adam_gru_14_gru_cell_14_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_gru_14_gru_cell_14_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp4assignvariableop_36_adam_gru_15_gru_cell_15_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp>assignvariableop_37_adam_gru_15_gru_cell_15_recurrent_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_gru_15_gru_cell_15_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39f
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_40?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
while_cond_697752
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_697752___redundant_placeholder04
0while_while_cond_697752___redundant_placeholder14
0while_while_cond_697752___redundant_placeholder24
0while_while_cond_697752___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
gru_15_while_cond_699907*
&gru_15_while_gru_15_while_loop_counter0
,gru_15_while_gru_15_while_maximum_iterations
gru_15_while_placeholder
gru_15_while_placeholder_1
gru_15_while_placeholder_2,
(gru_15_while_less_gru_15_strided_slice_1B
>gru_15_while_gru_15_while_cond_699907___redundant_placeholder0B
>gru_15_while_gru_15_while_cond_699907___redundant_placeholder1B
>gru_15_while_gru_15_while_cond_699907___redundant_placeholder2B
>gru_15_while_gru_15_while_cond_699907___redundant_placeholder3
gru_15_while_identity
?
gru_15/while/LessLessgru_15_while_placeholder(gru_15_while_less_gru_15_strided_slice_1*
T0*
_output_shapes
: 2
gru_15/while/Lessr
gru_15/while/IdentityIdentitygru_15/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_15/while/Identity"7
gru_15_while_identitygru_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_gru_15_layer_call_fn_701809

inputs
unknown:	?
	unknown_0:
??
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_15_layer_call_and_return_conditional_losses_6989352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%sequential_7_gru_15_while_cond_697332D
@sequential_7_gru_15_while_sequential_7_gru_15_while_loop_counterJ
Fsequential_7_gru_15_while_sequential_7_gru_15_while_maximum_iterations)
%sequential_7_gru_15_while_placeholder+
'sequential_7_gru_15_while_placeholder_1+
'sequential_7_gru_15_while_placeholder_2F
Bsequential_7_gru_15_while_less_sequential_7_gru_15_strided_slice_1\
Xsequential_7_gru_15_while_sequential_7_gru_15_while_cond_697332___redundant_placeholder0\
Xsequential_7_gru_15_while_sequential_7_gru_15_while_cond_697332___redundant_placeholder1\
Xsequential_7_gru_15_while_sequential_7_gru_15_while_cond_697332___redundant_placeholder2\
Xsequential_7_gru_15_while_sequential_7_gru_15_while_cond_697332___redundant_placeholder3&
"sequential_7_gru_15_while_identity
?
sequential_7/gru_15/while/LessLess%sequential_7_gru_15_while_placeholderBsequential_7_gru_15_while_less_sequential_7_gru_15_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_7/gru_15/while/Less?
"sequential_7/gru_15/while/IdentityIdentity"sequential_7/gru_15/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_7/gru_15/while/Identity"Q
"sequential_7_gru_15_while_identity+sequential_7/gru_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_697547

inputs

states*
readvariableop_resource:	?1
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_698256

inputs

states*
readvariableop_resource:	?2
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:??????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?"
?
H__inference_sequential_7_layer_call_and_return_conditional_losses_699630
gru_14_input 
gru_14_699602:	? 
gru_14_699604:	?!
gru_14_699606:
?? 
gru_15_699610:	?!
gru_15_699612:
??!
gru_15_699614:
??#
dense_14_699618:
??
dense_14_699620:	?"
dense_15_699624:	?
dense_15_699626:
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?gru_14/StatefulPartitionedCall?gru_15/StatefulPartitionedCall?
gru_14/StatefulPartitionedCallStatefulPartitionedCallgru_14_inputgru_14_699602gru_14_699604gru_14_699606*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_14_layer_call_and_return_conditional_losses_6987682 
gru_14/StatefulPartitionedCall?
dropout_21/PartitionedCallPartitionedCall'gru_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_6987812
dropout_21/PartitionedCall?
gru_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0gru_15_699610gru_15_699612gru_15_699614*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_15_layer_call_and_return_conditional_losses_6989352 
gru_15/StatefulPartitionedCall?
dropout_22/PartitionedCallPartitionedCall'gru_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6989482
dropout_22/PartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_14_699618dense_14_699620*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_6989812"
 dense_14/StatefulPartitionedCall?
dropout_23/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_6989922
dropout_23/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_15_699624dense_15_699626*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_6990242"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall^gru_14/StatefulPartitionedCall^gru_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2@
gru_14/StatefulPartitionedCallgru_14/StatefulPartitionedCall2@
gru_15/StatefulPartitionedCallgru_15/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_14_input
?X
?
B__inference_gru_15_layer_call_and_return_conditional_losses_701623

inputs6
#gru_cell_15_readvariableop_resource:	?>
*gru_cell_15_matmul_readvariableop_resource:
??@
,gru_cell_15_matmul_1_readvariableop_resource:
??
identity??!gru_cell_15/MatMul/ReadVariableOp?#gru_cell_15/MatMul_1/ReadVariableOp?gru_cell_15/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_15/ReadVariableOpReadVariableOp#gru_cell_15_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_15/ReadVariableOp?
gru_cell_15/unstackUnpack"gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_15/unstack?
!gru_cell_15/MatMul/ReadVariableOpReadVariableOp*gru_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_15/MatMul/ReadVariableOp?
gru_cell_15/MatMulMatMulstrided_slice_2:output:0)gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul?
gru_cell_15/BiasAddBiasAddgru_cell_15/MatMul:product:0gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd?
gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split/split_dim?
gru_cell_15/splitSplit$gru_cell_15/split/split_dim:output:0gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split?
#gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_15/MatMul_1/ReadVariableOp?
gru_cell_15/MatMul_1MatMulzeros:output:0+gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul_1?
gru_cell_15/BiasAdd_1BiasAddgru_cell_15/MatMul_1:product:0gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd_1{
gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_15/Const?
gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split_1/split_dim?
gru_cell_15/split_1SplitVgru_cell_15/BiasAdd_1:output:0gru_cell_15/Const:output:0&gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split_1?
gru_cell_15/addAddV2gru_cell_15/split:output:0gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add}
gru_cell_15/SigmoidSigmoidgru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid?
gru_cell_15/add_1AddV2gru_cell_15/split:output:1gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_1?
gru_cell_15/Sigmoid_1Sigmoidgru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid_1?
gru_cell_15/mulMulgru_cell_15/Sigmoid_1:y:0gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul?
gru_cell_15/add_2AddV2gru_cell_15/split:output:2gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_2v
gru_cell_15/ReluRelugru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Relu?
gru_cell_15/mul_1Mulgru_cell_15/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_1k
gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_15/sub/x?
gru_cell_15/subSubgru_cell_15/sub/x:output:0gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/sub?
gru_cell_15/mul_2Mulgru_cell_15/sub:z:0gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_2?
gru_cell_15/add_3AddV2gru_cell_15/mul_1:z:0gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_15_readvariableop_resource*gru_cell_15_matmul_readvariableop_resource,gru_cell_15_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_701534*
condR
while_cond_701533*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp"^gru_cell_15/MatMul/ReadVariableOp$^gru_cell_15/MatMul_1/ReadVariableOp^gru_cell_15/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_15/MatMul/ReadVariableOp!gru_cell_15/MatMul/ReadVariableOp2J
#gru_cell_15/MatMul_1/ReadVariableOp#gru_cell_15/MatMul_1/ReadVariableOp28
gru_cell_15/ReadVariableOpgru_cell_15/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?E
?
while_body_701534
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_15_readvariableop_resource_0:	?F
2while_gru_cell_15_matmul_readvariableop_resource_0:
??H
4while_gru_cell_15_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_15_readvariableop_resource:	?D
0while_gru_cell_15_matmul_readvariableop_resource:
??F
2while_gru_cell_15_matmul_1_readvariableop_resource:
????'while/gru_cell_15/MatMul/ReadVariableOp?)while/gru_cell_15/MatMul_1/ReadVariableOp? while/gru_cell_15/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_15/ReadVariableOpReadVariableOp+while_gru_cell_15_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_15/ReadVariableOp?
while/gru_cell_15/unstackUnpack(while/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_15/unstack?
'while/gru_cell_15/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_15/MatMul/ReadVariableOp?
while/gru_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul?
while/gru_cell_15/BiasAddBiasAdd"while/gru_cell_15/MatMul:product:0"while/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd?
!while/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_15/split/split_dim?
while/gru_cell_15/splitSplit*while/gru_cell_15/split/split_dim:output:0"while/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split?
)while/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_15/MatMul_1/ReadVariableOp?
while/gru_cell_15/MatMul_1MatMulwhile_placeholder_21while/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul_1?
while/gru_cell_15/BiasAdd_1BiasAdd$while/gru_cell_15/MatMul_1:product:0"while/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd_1?
while/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_15/Const?
#while/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_15/split_1/split_dim?
while/gru_cell_15/split_1SplitV$while/gru_cell_15/BiasAdd_1:output:0 while/gru_cell_15/Const:output:0,while/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split_1?
while/gru_cell_15/addAddV2 while/gru_cell_15/split:output:0"while/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add?
while/gru_cell_15/SigmoidSigmoidwhile/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid?
while/gru_cell_15/add_1AddV2 while/gru_cell_15/split:output:1"while/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_1?
while/gru_cell_15/Sigmoid_1Sigmoidwhile/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid_1?
while/gru_cell_15/mulMulwhile/gru_cell_15/Sigmoid_1:y:0"while/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul?
while/gru_cell_15/add_2AddV2 while/gru_cell_15/split:output:2while/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_2?
while/gru_cell_15/ReluReluwhile/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Relu?
while/gru_cell_15/mul_1Mulwhile/gru_cell_15/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_1w
while/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_15/sub/x?
while/gru_cell_15/subSub while/gru_cell_15/sub/x:output:0while/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/sub?
while/gru_cell_15/mul_2Mulwhile/gru_cell_15/sub:z:0$while/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_2?
while/gru_cell_15/add_3AddV2while/gru_cell_15/mul_1:z:0while/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_15/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_15/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_15/MatMul/ReadVariableOp*^while/gru_cell_15/MatMul_1/ReadVariableOp!^while/gru_cell_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_15_matmul_1_readvariableop_resource4while_gru_cell_15_matmul_1_readvariableop_resource_0"f
0while_gru_cell_15_matmul_readvariableop_resource2while_gru_cell_15_matmul_readvariableop_resource_0"X
)while_gru_cell_15_readvariableop_resource+while_gru_cell_15_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_15/MatMul/ReadVariableOp'while/gru_cell_15/MatMul/ReadVariableOp2V
)while/gru_cell_15/MatMul_1/ReadVariableOp)while/gru_cell_15/MatMul_1/ReadVariableOp2D
 while/gru_cell_15/ReadVariableOp while/gru_cell_15/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_702031

inputs
states_0*
readvariableop_resource:	?1
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?Y
?
B__inference_gru_15_layer_call_and_return_conditional_losses_701470
inputs_06
#gru_cell_15_readvariableop_resource:	?>
*gru_cell_15_matmul_readvariableop_resource:
??@
,gru_cell_15_matmul_1_readvariableop_resource:
??
identity??!gru_cell_15/MatMul/ReadVariableOp?#gru_cell_15/MatMul_1/ReadVariableOp?gru_cell_15/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_15/ReadVariableOpReadVariableOp#gru_cell_15_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_15/ReadVariableOp?
gru_cell_15/unstackUnpack"gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_15/unstack?
!gru_cell_15/MatMul/ReadVariableOpReadVariableOp*gru_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_15/MatMul/ReadVariableOp?
gru_cell_15/MatMulMatMulstrided_slice_2:output:0)gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul?
gru_cell_15/BiasAddBiasAddgru_cell_15/MatMul:product:0gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd?
gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split/split_dim?
gru_cell_15/splitSplit$gru_cell_15/split/split_dim:output:0gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split?
#gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_15/MatMul_1/ReadVariableOp?
gru_cell_15/MatMul_1MatMulzeros:output:0+gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul_1?
gru_cell_15/BiasAdd_1BiasAddgru_cell_15/MatMul_1:product:0gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd_1{
gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_15/Const?
gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split_1/split_dim?
gru_cell_15/split_1SplitVgru_cell_15/BiasAdd_1:output:0gru_cell_15/Const:output:0&gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split_1?
gru_cell_15/addAddV2gru_cell_15/split:output:0gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add}
gru_cell_15/SigmoidSigmoidgru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid?
gru_cell_15/add_1AddV2gru_cell_15/split:output:1gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_1?
gru_cell_15/Sigmoid_1Sigmoidgru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid_1?
gru_cell_15/mulMulgru_cell_15/Sigmoid_1:y:0gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul?
gru_cell_15/add_2AddV2gru_cell_15/split:output:2gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_2v
gru_cell_15/ReluRelugru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Relu?
gru_cell_15/mul_1Mulgru_cell_15/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_1k
gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_15/sub/x?
gru_cell_15/subSubgru_cell_15/sub/x:output:0gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/sub?
gru_cell_15/mul_2Mulgru_cell_15/sub:z:0gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_2?
gru_cell_15/add_3AddV2gru_cell_15/mul_1:z:0gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_15_readvariableop_resource*gru_cell_15_matmul_readvariableop_resource,gru_cell_15_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_701381*
condR
while_cond_701380*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp"^gru_cell_15/MatMul/ReadVariableOp$^gru_cell_15/MatMul_1/ReadVariableOp^gru_cell_15/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2F
!gru_cell_15/MatMul/ReadVariableOp!gru_cell_15/MatMul/ReadVariableOp2J
#gru_cell_15/MatMul_1/ReadVariableOp#gru_cell_15/MatMul_1/ReadVariableOp28
gru_cell_15/ReadVariableOpgru_cell_15/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
-__inference_sequential_7_layer_call_fn_699054
gru_14_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_6990312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_14_input
?
?
'__inference_gru_15_layer_call_fn_701820

inputs
unknown:	?
	unknown_0:
??
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_15_layer_call_and_return_conditional_losses_6992862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
D__inference_dense_14_layer_call_and_return_conditional_losses_698981

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_698318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_698318___redundant_placeholder04
0while_while_cond_698318___redundant_placeholder14
0while_while_cond_698318___redundant_placeholder24
0while_while_cond_698318___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_701380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_701380___redundant_placeholder04
0while_while_cond_701380___redundant_placeholder14
0while_while_cond_701380___redundant_placeholder24
0while_while_cond_701380___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_698992

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?&
?
H__inference_sequential_7_layer_call_and_return_conditional_losses_699551

inputs 
gru_14_699523:	? 
gru_14_699525:	?!
gru_14_699527:
?? 
gru_15_699531:	?!
gru_15_699533:
??!
gru_15_699535:
??#
dense_14_699539:
??
dense_14_699541:	?"
dense_15_699545:	?
dense_15_699547:
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?"dropout_21/StatefulPartitionedCall?"dropout_22/StatefulPartitionedCall?"dropout_23/StatefulPartitionedCall?gru_14/StatefulPartitionedCall?gru_15/StatefulPartitionedCall?
gru_14/StatefulPartitionedCallStatefulPartitionedCallinputsgru_14_699523gru_14_699525gru_14_699527*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_14_layer_call_and_return_conditional_losses_6994842 
gru_14/StatefulPartitionedCall?
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall'gru_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_6993152$
"dropout_21/StatefulPartitionedCall?
gru_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0gru_15_699531gru_15_699533gru_15_699535*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_15_layer_call_and_return_conditional_losses_6992862 
gru_15/StatefulPartitionedCall?
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall'gru_15/StatefulPartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6991172$
"dropout_22/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_14_699539dense_14_699541*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_6989812"
 dense_14/StatefulPartitionedCall?
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_6990842$
"dropout_23/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_15_699545dense_15_699547*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_6990242"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall^gru_14/StatefulPartitionedCall^gru_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2@
gru_14/StatefulPartitionedCallgru_14/StatefulPartitionedCall2@
gru_15/StatefulPartitionedCallgru_15/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
while_body_701004
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_14_readvariableop_resource_0:	?E
2while_gru_cell_14_matmul_readvariableop_resource_0:	?H
4while_gru_cell_14_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_14_readvariableop_resource:	?C
0while_gru_cell_14_matmul_readvariableop_resource:	?F
2while_gru_cell_14_matmul_1_readvariableop_resource:
????'while/gru_cell_14/MatMul/ReadVariableOp?)while/gru_cell_14/MatMul_1/ReadVariableOp? while/gru_cell_14/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_14/ReadVariableOpReadVariableOp+while_gru_cell_14_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_14/ReadVariableOp?
while/gru_cell_14/unstackUnpack(while/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_14/unstack?
'while/gru_cell_14/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_14/MatMul/ReadVariableOp?
while/gru_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul?
while/gru_cell_14/BiasAddBiasAdd"while/gru_cell_14/MatMul:product:0"while/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd?
!while/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_14/split/split_dim?
while/gru_cell_14/splitSplit*while/gru_cell_14/split/split_dim:output:0"while/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split?
)while/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_14/MatMul_1/ReadVariableOp?
while/gru_cell_14/MatMul_1MatMulwhile_placeholder_21while/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul_1?
while/gru_cell_14/BiasAdd_1BiasAdd$while/gru_cell_14/MatMul_1:product:0"while/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd_1?
while/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_14/Const?
#while/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_14/split_1/split_dim?
while/gru_cell_14/split_1SplitV$while/gru_cell_14/BiasAdd_1:output:0 while/gru_cell_14/Const:output:0,while/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split_1?
while/gru_cell_14/addAddV2 while/gru_cell_14/split:output:0"while/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add?
while/gru_cell_14/SigmoidSigmoidwhile/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid?
while/gru_cell_14/add_1AddV2 while/gru_cell_14/split:output:1"while/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_1?
while/gru_cell_14/Sigmoid_1Sigmoidwhile/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid_1?
while/gru_cell_14/mulMulwhile/gru_cell_14/Sigmoid_1:y:0"while/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul?
while/gru_cell_14/add_2AddV2 while/gru_cell_14/split:output:2while/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_2?
while/gru_cell_14/ReluReluwhile/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Relu?
while/gru_cell_14/mul_1Mulwhile/gru_cell_14/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_1w
while/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_14/sub/x?
while/gru_cell_14/subSub while/gru_cell_14/sub/x:output:0while/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/sub?
while/gru_cell_14/mul_2Mulwhile/gru_cell_14/sub:z:0$while/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_2?
while/gru_cell_14/add_3AddV2while/gru_cell_14/mul_1:z:0while/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_14/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_14/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_14/MatMul/ReadVariableOp*^while/gru_cell_14/MatMul_1/ReadVariableOp!^while/gru_cell_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_14_matmul_1_readvariableop_resource4while_gru_cell_14_matmul_1_readvariableop_resource_0"f
0while_gru_cell_14_matmul_readvariableop_resource2while_gru_cell_14_matmul_readvariableop_resource_0"X
)while_gru_cell_14_readvariableop_resource+while_gru_cell_14_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_14/MatMul/ReadVariableOp'while/gru_cell_14/MatMul/ReadVariableOp2V
)while/gru_cell_14/MatMul_1/ReadVariableOp)while/gru_cell_14/MatMul_1/ReadVariableOp2D
 while/gru_cell_14/ReadVariableOp while/gru_cell_14/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?X
?
B__inference_gru_14_layer_call_and_return_conditional_losses_701093

inputs6
#gru_cell_14_readvariableop_resource:	?=
*gru_cell_14_matmul_readvariableop_resource:	?@
,gru_cell_14_matmul_1_readvariableop_resource:
??
identity??!gru_cell_14/MatMul/ReadVariableOp?#gru_cell_14/MatMul_1/ReadVariableOp?gru_cell_14/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_14/ReadVariableOpReadVariableOp#gru_cell_14_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_14/ReadVariableOp?
gru_cell_14/unstackUnpack"gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_14/unstack?
!gru_cell_14/MatMul/ReadVariableOpReadVariableOp*gru_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_14/MatMul/ReadVariableOp?
gru_cell_14/MatMulMatMulstrided_slice_2:output:0)gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul?
gru_cell_14/BiasAddBiasAddgru_cell_14/MatMul:product:0gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd?
gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split/split_dim?
gru_cell_14/splitSplit$gru_cell_14/split/split_dim:output:0gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split?
#gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_14/MatMul_1/ReadVariableOp?
gru_cell_14/MatMul_1MatMulzeros:output:0+gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul_1?
gru_cell_14/BiasAdd_1BiasAddgru_cell_14/MatMul_1:product:0gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd_1{
gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_14/Const?
gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split_1/split_dim?
gru_cell_14/split_1SplitVgru_cell_14/BiasAdd_1:output:0gru_cell_14/Const:output:0&gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split_1?
gru_cell_14/addAddV2gru_cell_14/split:output:0gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add}
gru_cell_14/SigmoidSigmoidgru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid?
gru_cell_14/add_1AddV2gru_cell_14/split:output:1gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_1?
gru_cell_14/Sigmoid_1Sigmoidgru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid_1?
gru_cell_14/mulMulgru_cell_14/Sigmoid_1:y:0gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul?
gru_cell_14/add_2AddV2gru_cell_14/split:output:2gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_2v
gru_cell_14/ReluRelugru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Relu?
gru_cell_14/mul_1Mulgru_cell_14/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_1k
gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_14/sub/x?
gru_cell_14/subSubgru_cell_14/sub/x:output:0gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/sub?
gru_cell_14/mul_2Mulgru_cell_14/sub:z:0gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_2?
gru_cell_14/add_3AddV2gru_cell_14/mul_1:z:0gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_14_readvariableop_resource*gru_cell_14_matmul_readvariableop_resource,gru_cell_14_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_701004*
condR
while_cond_701003*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp"^gru_cell_14/MatMul/ReadVariableOp$^gru_cell_14/MatMul_1/ReadVariableOp^gru_cell_14/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_14/MatMul/ReadVariableOp!gru_cell_14/MatMul/ReadVariableOp2J
#gru_cell_14/MatMul_1/ReadVariableOp#gru_cell_14/MatMul_1/ReadVariableOp28
gru_cell_14/ReadVariableOpgru_cell_14/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ؒ
?	
H__inference_sequential_7_layer_call_and_return_conditional_losses_700052

inputs=
*gru_14_gru_cell_14_readvariableop_resource:	?D
1gru_14_gru_cell_14_matmul_readvariableop_resource:	?G
3gru_14_gru_cell_14_matmul_1_readvariableop_resource:
??=
*gru_15_gru_cell_15_readvariableop_resource:	?E
1gru_15_gru_cell_15_matmul_readvariableop_resource:
??G
3gru_15_gru_cell_15_matmul_1_readvariableop_resource:
??>
*dense_14_tensordot_readvariableop_resource:
??7
(dense_14_biasadd_readvariableop_resource:	?=
*dense_15_tensordot_readvariableop_resource:	?6
(dense_15_biasadd_readvariableop_resource:
identity??dense_14/BiasAdd/ReadVariableOp?!dense_14/Tensordot/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?!dense_15/Tensordot/ReadVariableOp?(gru_14/gru_cell_14/MatMul/ReadVariableOp?*gru_14/gru_cell_14/MatMul_1/ReadVariableOp?!gru_14/gru_cell_14/ReadVariableOp?gru_14/while?(gru_15/gru_cell_15/MatMul/ReadVariableOp?*gru_15/gru_cell_15/MatMul_1/ReadVariableOp?!gru_15/gru_cell_15/ReadVariableOp?gru_15/whileR
gru_14/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_14/Shape?
gru_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_14/strided_slice/stack?
gru_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_14/strided_slice/stack_1?
gru_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_14/strided_slice/stack_2?
gru_14/strided_sliceStridedSlicegru_14/Shape:output:0#gru_14/strided_slice/stack:output:0%gru_14/strided_slice/stack_1:output:0%gru_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_14/strided_sliceq
gru_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_14/zeros/packed/1?
gru_14/zeros/packedPackgru_14/strided_slice:output:0gru_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_14/zeros/packedm
gru_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_14/zeros/Const?
gru_14/zerosFillgru_14/zeros/packed:output:0gru_14/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_14/zeros?
gru_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_14/transpose/perm?
gru_14/transpose	Transposeinputsgru_14/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
gru_14/transposed
gru_14/Shape_1Shapegru_14/transpose:y:0*
T0*
_output_shapes
:2
gru_14/Shape_1?
gru_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_14/strided_slice_1/stack?
gru_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_14/strided_slice_1/stack_1?
gru_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_14/strided_slice_1/stack_2?
gru_14/strided_slice_1StridedSlicegru_14/Shape_1:output:0%gru_14/strided_slice_1/stack:output:0'gru_14/strided_slice_1/stack_1:output:0'gru_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_14/strided_slice_1?
"gru_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_14/TensorArrayV2/element_shape?
gru_14/TensorArrayV2TensorListReserve+gru_14/TensorArrayV2/element_shape:output:0gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_14/TensorArrayV2?
<gru_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_14/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_14/transpose:y:0Egru_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_14/TensorArrayUnstack/TensorListFromTensor?
gru_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_14/strided_slice_2/stack?
gru_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_14/strided_slice_2/stack_1?
gru_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_14/strided_slice_2/stack_2?
gru_14/strided_slice_2StridedSlicegru_14/transpose:y:0%gru_14/strided_slice_2/stack:output:0'gru_14/strided_slice_2/stack_1:output:0'gru_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_14/strided_slice_2?
!gru_14/gru_cell_14/ReadVariableOpReadVariableOp*gru_14_gru_cell_14_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_14/gru_cell_14/ReadVariableOp?
gru_14/gru_cell_14/unstackUnpack)gru_14/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_14/gru_cell_14/unstack?
(gru_14/gru_cell_14/MatMul/ReadVariableOpReadVariableOp1gru_14_gru_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_14/gru_cell_14/MatMul/ReadVariableOp?
gru_14/gru_cell_14/MatMulMatMulgru_14/strided_slice_2:output:00gru_14/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/MatMul?
gru_14/gru_cell_14/BiasAddBiasAdd#gru_14/gru_cell_14/MatMul:product:0#gru_14/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/BiasAdd?
"gru_14/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_14/gru_cell_14/split/split_dim?
gru_14/gru_cell_14/splitSplit+gru_14/gru_cell_14/split/split_dim:output:0#gru_14/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_14/gru_cell_14/split?
*gru_14/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp3gru_14_gru_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_14/gru_cell_14/MatMul_1/ReadVariableOp?
gru_14/gru_cell_14/MatMul_1MatMulgru_14/zeros:output:02gru_14/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/MatMul_1?
gru_14/gru_cell_14/BiasAdd_1BiasAdd%gru_14/gru_cell_14/MatMul_1:product:0#gru_14/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/BiasAdd_1?
gru_14/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_14/gru_cell_14/Const?
$gru_14/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_14/gru_cell_14/split_1/split_dim?
gru_14/gru_cell_14/split_1SplitV%gru_14/gru_cell_14/BiasAdd_1:output:0!gru_14/gru_cell_14/Const:output:0-gru_14/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_14/gru_cell_14/split_1?
gru_14/gru_cell_14/addAddV2!gru_14/gru_cell_14/split:output:0#gru_14/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/add?
gru_14/gru_cell_14/SigmoidSigmoidgru_14/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/Sigmoid?
gru_14/gru_cell_14/add_1AddV2!gru_14/gru_cell_14/split:output:1#gru_14/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/add_1?
gru_14/gru_cell_14/Sigmoid_1Sigmoidgru_14/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/Sigmoid_1?
gru_14/gru_cell_14/mulMul gru_14/gru_cell_14/Sigmoid_1:y:0#gru_14/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/mul?
gru_14/gru_cell_14/add_2AddV2!gru_14/gru_cell_14/split:output:2gru_14/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/add_2?
gru_14/gru_cell_14/ReluRelugru_14/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/Relu?
gru_14/gru_cell_14/mul_1Mulgru_14/gru_cell_14/Sigmoid:y:0gru_14/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/mul_1y
gru_14/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_14/gru_cell_14/sub/x?
gru_14/gru_cell_14/subSub!gru_14/gru_cell_14/sub/x:output:0gru_14/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/sub?
gru_14/gru_cell_14/mul_2Mulgru_14/gru_cell_14/sub:z:0%gru_14/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/mul_2?
gru_14/gru_cell_14/add_3AddV2gru_14/gru_cell_14/mul_1:z:0gru_14/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_14/gru_cell_14/add_3?
$gru_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$gru_14/TensorArrayV2_1/element_shape?
gru_14/TensorArrayV2_1TensorListReserve-gru_14/TensorArrayV2_1/element_shape:output:0gru_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_14/TensorArrayV2_1\
gru_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_14/time?
gru_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_14/while/maximum_iterationsx
gru_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_14/while/loop_counter?
gru_14/whileWhile"gru_14/while/loop_counter:output:0(gru_14/while/maximum_iterations:output:0gru_14/time:output:0gru_14/TensorArrayV2_1:handle:0gru_14/zeros:output:0gru_14/strided_slice_1:output:0>gru_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_14_gru_cell_14_readvariableop_resource1gru_14_gru_cell_14_matmul_readvariableop_resource3gru_14_gru_cell_14_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *$
bodyR
gru_14_while_body_699758*$
condR
gru_14_while_cond_699757*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_14/while?
7gru_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7gru_14/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_14/TensorArrayV2Stack/TensorListStackTensorListStackgru_14/while:output:3@gru_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_14/TensorArrayV2Stack/TensorListStack?
gru_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_14/strided_slice_3/stack?
gru_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_14/strided_slice_3/stack_1?
gru_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_14/strided_slice_3/stack_2?
gru_14/strided_slice_3StridedSlice2gru_14/TensorArrayV2Stack/TensorListStack:tensor:0%gru_14/strided_slice_3/stack:output:0'gru_14/strided_slice_3/stack_1:output:0'gru_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_14/strided_slice_3?
gru_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_14/transpose_1/perm?
gru_14/transpose_1	Transpose2gru_14/TensorArrayV2Stack/TensorListStack:tensor:0 gru_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_14/transpose_1t
gru_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_14/runtime?
dropout_21/IdentityIdentitygru_14/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_21/Identityh
gru_15/ShapeShapedropout_21/Identity:output:0*
T0*
_output_shapes
:2
gru_15/Shape?
gru_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_15/strided_slice/stack?
gru_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_15/strided_slice/stack_1?
gru_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_15/strided_slice/stack_2?
gru_15/strided_sliceStridedSlicegru_15/Shape:output:0#gru_15/strided_slice/stack:output:0%gru_15/strided_slice/stack_1:output:0%gru_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_15/strided_sliceq
gru_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_15/zeros/packed/1?
gru_15/zeros/packedPackgru_15/strided_slice:output:0gru_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_15/zeros/packedm
gru_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_15/zeros/Const?
gru_15/zerosFillgru_15/zeros/packed:output:0gru_15/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_15/zeros?
gru_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_15/transpose/perm?
gru_15/transpose	Transposedropout_21/Identity:output:0gru_15/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_15/transposed
gru_15/Shape_1Shapegru_15/transpose:y:0*
T0*
_output_shapes
:2
gru_15/Shape_1?
gru_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_15/strided_slice_1/stack?
gru_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_15/strided_slice_1/stack_1?
gru_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_15/strided_slice_1/stack_2?
gru_15/strided_slice_1StridedSlicegru_15/Shape_1:output:0%gru_15/strided_slice_1/stack:output:0'gru_15/strided_slice_1/stack_1:output:0'gru_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_15/strided_slice_1?
"gru_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_15/TensorArrayV2/element_shape?
gru_15/TensorArrayV2TensorListReserve+gru_15/TensorArrayV2/element_shape:output:0gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_15/TensorArrayV2?
<gru_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_15/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_15/transpose:y:0Egru_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_15/TensorArrayUnstack/TensorListFromTensor?
gru_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_15/strided_slice_2/stack?
gru_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_15/strided_slice_2/stack_1?
gru_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_15/strided_slice_2/stack_2?
gru_15/strided_slice_2StridedSlicegru_15/transpose:y:0%gru_15/strided_slice_2/stack:output:0'gru_15/strided_slice_2/stack_1:output:0'gru_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_15/strided_slice_2?
!gru_15/gru_cell_15/ReadVariableOpReadVariableOp*gru_15_gru_cell_15_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_15/gru_cell_15/ReadVariableOp?
gru_15/gru_cell_15/unstackUnpack)gru_15/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_15/gru_cell_15/unstack?
(gru_15/gru_cell_15/MatMul/ReadVariableOpReadVariableOp1gru_15_gru_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(gru_15/gru_cell_15/MatMul/ReadVariableOp?
gru_15/gru_cell_15/MatMulMatMulgru_15/strided_slice_2:output:00gru_15/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/MatMul?
gru_15/gru_cell_15/BiasAddBiasAdd#gru_15/gru_cell_15/MatMul:product:0#gru_15/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/BiasAdd?
"gru_15/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_15/gru_cell_15/split/split_dim?
gru_15/gru_cell_15/splitSplit+gru_15/gru_cell_15/split/split_dim:output:0#gru_15/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_15/gru_cell_15/split?
*gru_15/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp3gru_15_gru_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_15/gru_cell_15/MatMul_1/ReadVariableOp?
gru_15/gru_cell_15/MatMul_1MatMulgru_15/zeros:output:02gru_15/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/MatMul_1?
gru_15/gru_cell_15/BiasAdd_1BiasAdd%gru_15/gru_cell_15/MatMul_1:product:0#gru_15/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/BiasAdd_1?
gru_15/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_15/gru_cell_15/Const?
$gru_15/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_15/gru_cell_15/split_1/split_dim?
gru_15/gru_cell_15/split_1SplitV%gru_15/gru_cell_15/BiasAdd_1:output:0!gru_15/gru_cell_15/Const:output:0-gru_15/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_15/gru_cell_15/split_1?
gru_15/gru_cell_15/addAddV2!gru_15/gru_cell_15/split:output:0#gru_15/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/add?
gru_15/gru_cell_15/SigmoidSigmoidgru_15/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/Sigmoid?
gru_15/gru_cell_15/add_1AddV2!gru_15/gru_cell_15/split:output:1#gru_15/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/add_1?
gru_15/gru_cell_15/Sigmoid_1Sigmoidgru_15/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/Sigmoid_1?
gru_15/gru_cell_15/mulMul gru_15/gru_cell_15/Sigmoid_1:y:0#gru_15/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/mul?
gru_15/gru_cell_15/add_2AddV2!gru_15/gru_cell_15/split:output:2gru_15/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/add_2?
gru_15/gru_cell_15/ReluRelugru_15/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/Relu?
gru_15/gru_cell_15/mul_1Mulgru_15/gru_cell_15/Sigmoid:y:0gru_15/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/mul_1y
gru_15/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_15/gru_cell_15/sub/x?
gru_15/gru_cell_15/subSub!gru_15/gru_cell_15/sub/x:output:0gru_15/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/sub?
gru_15/gru_cell_15/mul_2Mulgru_15/gru_cell_15/sub:z:0%gru_15/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/mul_2?
gru_15/gru_cell_15/add_3AddV2gru_15/gru_cell_15/mul_1:z:0gru_15/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_15/gru_cell_15/add_3?
$gru_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$gru_15/TensorArrayV2_1/element_shape?
gru_15/TensorArrayV2_1TensorListReserve-gru_15/TensorArrayV2_1/element_shape:output:0gru_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_15/TensorArrayV2_1\
gru_15/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_15/time?
gru_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_15/while/maximum_iterationsx
gru_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_15/while/loop_counter?
gru_15/whileWhile"gru_15/while/loop_counter:output:0(gru_15/while/maximum_iterations:output:0gru_15/time:output:0gru_15/TensorArrayV2_1:handle:0gru_15/zeros:output:0gru_15/strided_slice_1:output:0>gru_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_15_gru_cell_15_readvariableop_resource1gru_15_gru_cell_15_matmul_readvariableop_resource3gru_15_gru_cell_15_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *$
bodyR
gru_15_while_body_699908*$
condR
gru_15_while_cond_699907*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_15/while?
7gru_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7gru_15/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_15/TensorArrayV2Stack/TensorListStackTensorListStackgru_15/while:output:3@gru_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_15/TensorArrayV2Stack/TensorListStack?
gru_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_15/strided_slice_3/stack?
gru_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_15/strided_slice_3/stack_1?
gru_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_15/strided_slice_3/stack_2?
gru_15/strided_slice_3StridedSlice2gru_15/TensorArrayV2Stack/TensorListStack:tensor:0%gru_15/strided_slice_3/stack:output:0'gru_15/strided_slice_3/stack_1:output:0'gru_15/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_15/strided_slice_3?
gru_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_15/transpose_1/perm?
gru_15/transpose_1	Transpose2gru_15/TensorArrayV2Stack/TensorListStack:tensor:0 gru_15/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_15/transpose_1t
gru_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_15/runtime?
dropout_22/IdentityIdentitygru_15/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_22/Identity?
!dense_14/Tensordot/ReadVariableOpReadVariableOp*dense_14_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_14/Tensordot/ReadVariableOp|
dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_14/Tensordot/axes?
dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_14/Tensordot/free?
dense_14/Tensordot/ShapeShapedropout_22/Identity:output:0*
T0*
_output_shapes
:2
dense_14/Tensordot/Shape?
 dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_14/Tensordot/GatherV2/axis?
dense_14/Tensordot/GatherV2GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/free:output:0)dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_14/Tensordot/GatherV2?
"dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_14/Tensordot/GatherV2_1/axis?
dense_14/Tensordot/GatherV2_1GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/axes:output:0+dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_14/Tensordot/GatherV2_1~
dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_14/Tensordot/Const?
dense_14/Tensordot/ProdProd$dense_14/Tensordot/GatherV2:output:0!dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_14/Tensordot/Prod?
dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_14/Tensordot/Const_1?
dense_14/Tensordot/Prod_1Prod&dense_14/Tensordot/GatherV2_1:output:0#dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_14/Tensordot/Prod_1?
dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_14/Tensordot/concat/axis?
dense_14/Tensordot/concatConcatV2 dense_14/Tensordot/free:output:0 dense_14/Tensordot/axes:output:0'dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_14/Tensordot/concat?
dense_14/Tensordot/stackPack dense_14/Tensordot/Prod:output:0"dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_14/Tensordot/stack?
dense_14/Tensordot/transpose	Transposedropout_22/Identity:output:0"dense_14/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_14/Tensordot/transpose?
dense_14/Tensordot/ReshapeReshape dense_14/Tensordot/transpose:y:0!dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_14/Tensordot/Reshape?
dense_14/Tensordot/MatMulMatMul#dense_14/Tensordot/Reshape:output:0)dense_14/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_14/Tensordot/MatMul?
dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_14/Tensordot/Const_2?
 dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_14/Tensordot/concat_1/axis?
dense_14/Tensordot/concat_1ConcatV2$dense_14/Tensordot/GatherV2:output:0#dense_14/Tensordot/Const_2:output:0)dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_14/Tensordot/concat_1?
dense_14/TensordotReshape#dense_14/Tensordot/MatMul:product:0$dense_14/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_14/Tensordot?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/Tensordot:output:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_14/BiasAddx
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_14/Relu?
dropout_23/IdentityIdentitydense_14/Relu:activations:0*
T0*,
_output_shapes
:??????????2
dropout_23/Identity?
!dense_15/Tensordot/ReadVariableOpReadVariableOp*dense_15_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_15/Tensordot/ReadVariableOp|
dense_15/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_15/Tensordot/axes?
dense_15/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_15/Tensordot/free?
dense_15/Tensordot/ShapeShapedropout_23/Identity:output:0*
T0*
_output_shapes
:2
dense_15/Tensordot/Shape?
 dense_15/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_15/Tensordot/GatherV2/axis?
dense_15/Tensordot/GatherV2GatherV2!dense_15/Tensordot/Shape:output:0 dense_15/Tensordot/free:output:0)dense_15/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_15/Tensordot/GatherV2?
"dense_15/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_15/Tensordot/GatherV2_1/axis?
dense_15/Tensordot/GatherV2_1GatherV2!dense_15/Tensordot/Shape:output:0 dense_15/Tensordot/axes:output:0+dense_15/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_15/Tensordot/GatherV2_1~
dense_15/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_15/Tensordot/Const?
dense_15/Tensordot/ProdProd$dense_15/Tensordot/GatherV2:output:0!dense_15/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_15/Tensordot/Prod?
dense_15/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_15/Tensordot/Const_1?
dense_15/Tensordot/Prod_1Prod&dense_15/Tensordot/GatherV2_1:output:0#dense_15/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_15/Tensordot/Prod_1?
dense_15/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_15/Tensordot/concat/axis?
dense_15/Tensordot/concatConcatV2 dense_15/Tensordot/free:output:0 dense_15/Tensordot/axes:output:0'dense_15/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_15/Tensordot/concat?
dense_15/Tensordot/stackPack dense_15/Tensordot/Prod:output:0"dense_15/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_15/Tensordot/stack?
dense_15/Tensordot/transpose	Transposedropout_23/Identity:output:0"dense_15/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_15/Tensordot/transpose?
dense_15/Tensordot/ReshapeReshape dense_15/Tensordot/transpose:y:0!dense_15/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_15/Tensordot/Reshape?
dense_15/Tensordot/MatMulMatMul#dense_15/Tensordot/Reshape:output:0)dense_15/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_15/Tensordot/MatMul?
dense_15/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_15/Tensordot/Const_2?
 dense_15/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_15/Tensordot/concat_1/axis?
dense_15/Tensordot/concat_1ConcatV2$dense_15/Tensordot/GatherV2:output:0#dense_15/Tensordot/Const_2:output:0)dense_15/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_15/Tensordot/concat_1?
dense_15/TensordotReshape#dense_15/Tensordot/MatMul:product:0$dense_15/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_15/Tensordot?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/Tensordot:output:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_15/BiasAddx
IdentityIdentitydense_15/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp"^dense_14/Tensordot/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp"^dense_15/Tensordot/ReadVariableOp)^gru_14/gru_cell_14/MatMul/ReadVariableOp+^gru_14/gru_cell_14/MatMul_1/ReadVariableOp"^gru_14/gru_cell_14/ReadVariableOp^gru_14/while)^gru_15/gru_cell_15/MatMul/ReadVariableOp+^gru_15/gru_cell_15/MatMul_1/ReadVariableOp"^gru_15/gru_cell_15/ReadVariableOp^gru_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/Tensordot/ReadVariableOp!dense_14/Tensordot/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2F
!dense_15/Tensordot/ReadVariableOp!dense_15/Tensordot/ReadVariableOp2T
(gru_14/gru_cell_14/MatMul/ReadVariableOp(gru_14/gru_cell_14/MatMul/ReadVariableOp2X
*gru_14/gru_cell_14/MatMul_1/ReadVariableOp*gru_14/gru_cell_14/MatMul_1/ReadVariableOp2F
!gru_14/gru_cell_14/ReadVariableOp!gru_14/gru_cell_14/ReadVariableOp2
gru_14/whilegru_14/while2T
(gru_15/gru_cell_15/MatMul/ReadVariableOp(gru_15/gru_cell_15/MatMul/ReadVariableOp2X
*gru_15/gru_cell_15/MatMul_1/ReadVariableOp*gru_15/gru_cell_15/MatMul_1/ReadVariableOp2F
!gru_15/gru_cell_15/ReadVariableOp!gru_15/gru_cell_15/ReadVariableOp2
gru_15/whilegru_15/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_gru_14_layer_call_fn_701137

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_14_layer_call_and_return_conditional_losses_6994842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?V
?
__inference__traced_save_702305
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_gru_14_gru_cell_14_kernel_read_readvariableopB
>savev2_gru_14_gru_cell_14_recurrent_kernel_read_readvariableop6
2savev2_gru_14_gru_cell_14_bias_read_readvariableop8
4savev2_gru_15_gru_cell_15_kernel_read_readvariableopB
>savev2_gru_15_gru_cell_15_recurrent_kernel_read_readvariableop6
2savev2_gru_15_gru_cell_15_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableop?
;savev2_adam_gru_14_gru_cell_14_kernel_m_read_readvariableopI
Esavev2_adam_gru_14_gru_cell_14_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_14_gru_cell_14_bias_m_read_readvariableop?
;savev2_adam_gru_15_gru_cell_15_kernel_m_read_readvariableopI
Esavev2_adam_gru_15_gru_cell_15_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_15_gru_cell_15_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableop?
;savev2_adam_gru_14_gru_cell_14_kernel_v_read_readvariableopI
Esavev2_adam_gru_14_gru_cell_14_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_14_gru_cell_14_bias_v_read_readvariableop?
;savev2_adam_gru_15_gru_cell_15_kernel_v_read_readvariableopI
Esavev2_adam_gru_15_gru_cell_15_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_15_gru_cell_15_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_gru_14_gru_cell_14_kernel_read_readvariableop>savev2_gru_14_gru_cell_14_recurrent_kernel_read_readvariableop2savev2_gru_14_gru_cell_14_bias_read_readvariableop4savev2_gru_15_gru_cell_15_kernel_read_readvariableop>savev2_gru_15_gru_cell_15_recurrent_kernel_read_readvariableop2savev2_gru_15_gru_cell_15_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop;savev2_adam_gru_14_gru_cell_14_kernel_m_read_readvariableopEsavev2_adam_gru_14_gru_cell_14_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_14_gru_cell_14_bias_m_read_readvariableop;savev2_adam_gru_15_gru_cell_15_kernel_m_read_readvariableopEsavev2_adam_gru_15_gru_cell_15_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_15_gru_cell_15_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableop;savev2_adam_gru_14_gru_cell_14_kernel_v_read_readvariableopEsavev2_adam_gru_14_gru_cell_14_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_14_gru_cell_14_bias_v_read_readvariableop;savev2_adam_gru_15_gru_cell_15_kernel_v_read_readvariableopEsavev2_adam_gru_15_gru_cell_15_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_15_gru_cell_15_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:	?:: : : : : :	?:
??:	?:
??:
??:	?: : : : :
??:?:	?::	?:
??:	?:
??:
??:	?:
??:?:	?::	?:
??:	?:
??:
??:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?: !

_output_shapes
::%"!

_output_shapes
:	?:&#"
 
_output_shapes
:
??:%$!

_output_shapes
:	?:&%"
 
_output_shapes
:
??:&&"
 
_output_shapes
:
??:%'!

_output_shapes
:	?:(

_output_shapes
: 
?
?
-__inference_sequential_7_layer_call_fn_699599
gru_14_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_6995512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_14_input
?
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_701825

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?X
?
B__inference_gru_14_layer_call_and_return_conditional_losses_698768

inputs6
#gru_cell_14_readvariableop_resource:	?=
*gru_cell_14_matmul_readvariableop_resource:	?@
,gru_cell_14_matmul_1_readvariableop_resource:
??
identity??!gru_cell_14/MatMul/ReadVariableOp?#gru_cell_14/MatMul_1/ReadVariableOp?gru_cell_14/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_14/ReadVariableOpReadVariableOp#gru_cell_14_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_14/ReadVariableOp?
gru_cell_14/unstackUnpack"gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_14/unstack?
!gru_cell_14/MatMul/ReadVariableOpReadVariableOp*gru_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_14/MatMul/ReadVariableOp?
gru_cell_14/MatMulMatMulstrided_slice_2:output:0)gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul?
gru_cell_14/BiasAddBiasAddgru_cell_14/MatMul:product:0gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd?
gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split/split_dim?
gru_cell_14/splitSplit$gru_cell_14/split/split_dim:output:0gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split?
#gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_14/MatMul_1/ReadVariableOp?
gru_cell_14/MatMul_1MatMulzeros:output:0+gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul_1?
gru_cell_14/BiasAdd_1BiasAddgru_cell_14/MatMul_1:product:0gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd_1{
gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_14/Const?
gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split_1/split_dim?
gru_cell_14/split_1SplitVgru_cell_14/BiasAdd_1:output:0gru_cell_14/Const:output:0&gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split_1?
gru_cell_14/addAddV2gru_cell_14/split:output:0gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add}
gru_cell_14/SigmoidSigmoidgru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid?
gru_cell_14/add_1AddV2gru_cell_14/split:output:1gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_1?
gru_cell_14/Sigmoid_1Sigmoidgru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid_1?
gru_cell_14/mulMulgru_cell_14/Sigmoid_1:y:0gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul?
gru_cell_14/add_2AddV2gru_cell_14/split:output:2gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_2v
gru_cell_14/ReluRelugru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Relu?
gru_cell_14/mul_1Mulgru_cell_14/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_1k
gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_14/sub/x?
gru_cell_14/subSubgru_cell_14/sub/x:output:0gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/sub?
gru_cell_14/mul_2Mulgru_cell_14/sub:z:0gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_2?
gru_cell_14/add_3AddV2gru_cell_14/mul_1:z:0gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_14_readvariableop_resource*gru_cell_14_matmul_readvariableop_resource,gru_cell_14_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_698679*
condR
while_cond_698678*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp"^gru_cell_14/MatMul/ReadVariableOp$^gru_cell_14/MatMul_1/ReadVariableOp^gru_cell_14/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_14/MatMul/ReadVariableOp!gru_cell_14/MatMul/ReadVariableOp2J
#gru_cell_14/MatMul_1/ReadVariableOp#gru_cell_14/MatMul_1/ReadVariableOp28
gru_cell_14/ReadVariableOpgru_cell_14/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_22_layer_call_and_return_conditional_losses_701837

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?P
?	
gru_14_while_body_699758*
&gru_14_while_gru_14_while_loop_counter0
,gru_14_while_gru_14_while_maximum_iterations
gru_14_while_placeholder
gru_14_while_placeholder_1
gru_14_while_placeholder_2)
%gru_14_while_gru_14_strided_slice_1_0e
agru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0E
2gru_14_while_gru_cell_14_readvariableop_resource_0:	?L
9gru_14_while_gru_cell_14_matmul_readvariableop_resource_0:	?O
;gru_14_while_gru_cell_14_matmul_1_readvariableop_resource_0:
??
gru_14_while_identity
gru_14_while_identity_1
gru_14_while_identity_2
gru_14_while_identity_3
gru_14_while_identity_4'
#gru_14_while_gru_14_strided_slice_1c
_gru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensorC
0gru_14_while_gru_cell_14_readvariableop_resource:	?J
7gru_14_while_gru_cell_14_matmul_readvariableop_resource:	?M
9gru_14_while_gru_cell_14_matmul_1_readvariableop_resource:
????.gru_14/while/gru_cell_14/MatMul/ReadVariableOp?0gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp?'gru_14/while/gru_cell_14/ReadVariableOp?
>gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0gru_14_while_placeholderGgru_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_14/while/TensorArrayV2Read/TensorListGetItem?
'gru_14/while/gru_cell_14/ReadVariableOpReadVariableOp2gru_14_while_gru_cell_14_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_14/while/gru_cell_14/ReadVariableOp?
 gru_14/while/gru_cell_14/unstackUnpack/gru_14/while/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_14/while/gru_cell_14/unstack?
.gru_14/while/gru_cell_14/MatMul/ReadVariableOpReadVariableOp9gru_14_while_gru_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_14/while/gru_cell_14/MatMul/ReadVariableOp?
gru_14/while/gru_cell_14/MatMulMatMul7gru_14/while/TensorArrayV2Read/TensorListGetItem:item:06gru_14/while/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_14/while/gru_cell_14/MatMul?
 gru_14/while/gru_cell_14/BiasAddBiasAdd)gru_14/while/gru_cell_14/MatMul:product:0)gru_14/while/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_14/while/gru_cell_14/BiasAdd?
(gru_14/while/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_14/while/gru_cell_14/split/split_dim?
gru_14/while/gru_cell_14/splitSplit1gru_14/while/gru_cell_14/split/split_dim:output:0)gru_14/while/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_14/while/gru_cell_14/split?
0gru_14/while/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp;gru_14_while_gru_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp?
!gru_14/while/gru_cell_14/MatMul_1MatMulgru_14_while_placeholder_28gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_14/while/gru_cell_14/MatMul_1?
"gru_14/while/gru_cell_14/BiasAdd_1BiasAdd+gru_14/while/gru_cell_14/MatMul_1:product:0)gru_14/while/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_14/while/gru_cell_14/BiasAdd_1?
gru_14/while/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2 
gru_14/while/gru_cell_14/Const?
*gru_14/while/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_14/while/gru_cell_14/split_1/split_dim?
 gru_14/while/gru_cell_14/split_1SplitV+gru_14/while/gru_cell_14/BiasAdd_1:output:0'gru_14/while/gru_cell_14/Const:output:03gru_14/while/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_14/while/gru_cell_14/split_1?
gru_14/while/gru_cell_14/addAddV2'gru_14/while/gru_cell_14/split:output:0)gru_14/while/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_14/while/gru_cell_14/add?
 gru_14/while/gru_cell_14/SigmoidSigmoid gru_14/while/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_14/while/gru_cell_14/Sigmoid?
gru_14/while/gru_cell_14/add_1AddV2'gru_14/while/gru_cell_14/split:output:1)gru_14/while/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_14/while/gru_cell_14/add_1?
"gru_14/while/gru_cell_14/Sigmoid_1Sigmoid"gru_14/while/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_14/while/gru_cell_14/Sigmoid_1?
gru_14/while/gru_cell_14/mulMul&gru_14/while/gru_cell_14/Sigmoid_1:y:0)gru_14/while/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_14/while/gru_cell_14/mul?
gru_14/while/gru_cell_14/add_2AddV2'gru_14/while/gru_cell_14/split:output:2 gru_14/while/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_14/while/gru_cell_14/add_2?
gru_14/while/gru_cell_14/ReluRelu"gru_14/while/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_14/while/gru_cell_14/Relu?
gru_14/while/gru_cell_14/mul_1Mul$gru_14/while/gru_cell_14/Sigmoid:y:0gru_14_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_14/while/gru_cell_14/mul_1?
gru_14/while/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_14/while/gru_cell_14/sub/x?
gru_14/while/gru_cell_14/subSub'gru_14/while/gru_cell_14/sub/x:output:0$gru_14/while/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_14/while/gru_cell_14/sub?
gru_14/while/gru_cell_14/mul_2Mul gru_14/while/gru_cell_14/sub:z:0+gru_14/while/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_14/while/gru_cell_14/mul_2?
gru_14/while/gru_cell_14/add_3AddV2"gru_14/while/gru_cell_14/mul_1:z:0"gru_14/while/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_14/while/gru_cell_14/add_3?
1gru_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_14_while_placeholder_1gru_14_while_placeholder"gru_14/while/gru_cell_14/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_14/while/TensorArrayV2Write/TensorListSetItemj
gru_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_14/while/add/y?
gru_14/while/addAddV2gru_14_while_placeholdergru_14/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_14/while/addn
gru_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_14/while/add_1/y?
gru_14/while/add_1AddV2&gru_14_while_gru_14_while_loop_countergru_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_14/while/add_1?
gru_14/while/IdentityIdentitygru_14/while/add_1:z:0^gru_14/while/NoOp*
T0*
_output_shapes
: 2
gru_14/while/Identity?
gru_14/while/Identity_1Identity,gru_14_while_gru_14_while_maximum_iterations^gru_14/while/NoOp*
T0*
_output_shapes
: 2
gru_14/while/Identity_1?
gru_14/while/Identity_2Identitygru_14/while/add:z:0^gru_14/while/NoOp*
T0*
_output_shapes
: 2
gru_14/while/Identity_2?
gru_14/while/Identity_3IdentityAgru_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_14/while/NoOp*
T0*
_output_shapes
: 2
gru_14/while/Identity_3?
gru_14/while/Identity_4Identity"gru_14/while/gru_cell_14/add_3:z:0^gru_14/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_14/while/Identity_4?
gru_14/while/NoOpNoOp/^gru_14/while/gru_cell_14/MatMul/ReadVariableOp1^gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp(^gru_14/while/gru_cell_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_14/while/NoOp"L
#gru_14_while_gru_14_strided_slice_1%gru_14_while_gru_14_strided_slice_1_0"x
9gru_14_while_gru_cell_14_matmul_1_readvariableop_resource;gru_14_while_gru_cell_14_matmul_1_readvariableop_resource_0"t
7gru_14_while_gru_cell_14_matmul_readvariableop_resource9gru_14_while_gru_cell_14_matmul_readvariableop_resource_0"f
0gru_14_while_gru_cell_14_readvariableop_resource2gru_14_while_gru_cell_14_readvariableop_resource_0"7
gru_14_while_identitygru_14/while/Identity:output:0";
gru_14_while_identity_1 gru_14/while/Identity_1:output:0";
gru_14_while_identity_2 gru_14/while/Identity_2:output:0";
gru_14_while_identity_3 gru_14/while/Identity_3:output:0";
gru_14_while_identity_4 gru_14/while/Identity_4:output:0"?
_gru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensoragru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_14/while/gru_cell_14/MatMul/ReadVariableOp.gru_14/while/gru_cell_14/MatMul/ReadVariableOp2d
0gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp0gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp2R
'gru_14/while/gru_cell_14/ReadVariableOp'gru_14/while/gru_cell_14/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_701686
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_701686___redundant_placeholder04
0while_while_cond_701686___redundant_placeholder14
0while_while_cond_701686___redundant_placeholder24
0while_while_cond_701686___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
'__inference_gru_14_layer_call_fn_701126

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_14_layer_call_and_return_conditional_losses_6987682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?P
?	
gru_14_while_body_700116*
&gru_14_while_gru_14_while_loop_counter0
,gru_14_while_gru_14_while_maximum_iterations
gru_14_while_placeholder
gru_14_while_placeholder_1
gru_14_while_placeholder_2)
%gru_14_while_gru_14_strided_slice_1_0e
agru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0E
2gru_14_while_gru_cell_14_readvariableop_resource_0:	?L
9gru_14_while_gru_cell_14_matmul_readvariableop_resource_0:	?O
;gru_14_while_gru_cell_14_matmul_1_readvariableop_resource_0:
??
gru_14_while_identity
gru_14_while_identity_1
gru_14_while_identity_2
gru_14_while_identity_3
gru_14_while_identity_4'
#gru_14_while_gru_14_strided_slice_1c
_gru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensorC
0gru_14_while_gru_cell_14_readvariableop_resource:	?J
7gru_14_while_gru_cell_14_matmul_readvariableop_resource:	?M
9gru_14_while_gru_cell_14_matmul_1_readvariableop_resource:
????.gru_14/while/gru_cell_14/MatMul/ReadVariableOp?0gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp?'gru_14/while/gru_cell_14/ReadVariableOp?
>gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_14/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0gru_14_while_placeholderGgru_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_14/while/TensorArrayV2Read/TensorListGetItem?
'gru_14/while/gru_cell_14/ReadVariableOpReadVariableOp2gru_14_while_gru_cell_14_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_14/while/gru_cell_14/ReadVariableOp?
 gru_14/while/gru_cell_14/unstackUnpack/gru_14/while/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_14/while/gru_cell_14/unstack?
.gru_14/while/gru_cell_14/MatMul/ReadVariableOpReadVariableOp9gru_14_while_gru_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_14/while/gru_cell_14/MatMul/ReadVariableOp?
gru_14/while/gru_cell_14/MatMulMatMul7gru_14/while/TensorArrayV2Read/TensorListGetItem:item:06gru_14/while/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_14/while/gru_cell_14/MatMul?
 gru_14/while/gru_cell_14/BiasAddBiasAdd)gru_14/while/gru_cell_14/MatMul:product:0)gru_14/while/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_14/while/gru_cell_14/BiasAdd?
(gru_14/while/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_14/while/gru_cell_14/split/split_dim?
gru_14/while/gru_cell_14/splitSplit1gru_14/while/gru_cell_14/split/split_dim:output:0)gru_14/while/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_14/while/gru_cell_14/split?
0gru_14/while/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp;gru_14_while_gru_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp?
!gru_14/while/gru_cell_14/MatMul_1MatMulgru_14_while_placeholder_28gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_14/while/gru_cell_14/MatMul_1?
"gru_14/while/gru_cell_14/BiasAdd_1BiasAdd+gru_14/while/gru_cell_14/MatMul_1:product:0)gru_14/while/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_14/while/gru_cell_14/BiasAdd_1?
gru_14/while/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2 
gru_14/while/gru_cell_14/Const?
*gru_14/while/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_14/while/gru_cell_14/split_1/split_dim?
 gru_14/while/gru_cell_14/split_1SplitV+gru_14/while/gru_cell_14/BiasAdd_1:output:0'gru_14/while/gru_cell_14/Const:output:03gru_14/while/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_14/while/gru_cell_14/split_1?
gru_14/while/gru_cell_14/addAddV2'gru_14/while/gru_cell_14/split:output:0)gru_14/while/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_14/while/gru_cell_14/add?
 gru_14/while/gru_cell_14/SigmoidSigmoid gru_14/while/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_14/while/gru_cell_14/Sigmoid?
gru_14/while/gru_cell_14/add_1AddV2'gru_14/while/gru_cell_14/split:output:1)gru_14/while/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_14/while/gru_cell_14/add_1?
"gru_14/while/gru_cell_14/Sigmoid_1Sigmoid"gru_14/while/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_14/while/gru_cell_14/Sigmoid_1?
gru_14/while/gru_cell_14/mulMul&gru_14/while/gru_cell_14/Sigmoid_1:y:0)gru_14/while/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_14/while/gru_cell_14/mul?
gru_14/while/gru_cell_14/add_2AddV2'gru_14/while/gru_cell_14/split:output:2 gru_14/while/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_14/while/gru_cell_14/add_2?
gru_14/while/gru_cell_14/ReluRelu"gru_14/while/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_14/while/gru_cell_14/Relu?
gru_14/while/gru_cell_14/mul_1Mul$gru_14/while/gru_cell_14/Sigmoid:y:0gru_14_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_14/while/gru_cell_14/mul_1?
gru_14/while/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_14/while/gru_cell_14/sub/x?
gru_14/while/gru_cell_14/subSub'gru_14/while/gru_cell_14/sub/x:output:0$gru_14/while/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_14/while/gru_cell_14/sub?
gru_14/while/gru_cell_14/mul_2Mul gru_14/while/gru_cell_14/sub:z:0+gru_14/while/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_14/while/gru_cell_14/mul_2?
gru_14/while/gru_cell_14/add_3AddV2"gru_14/while/gru_cell_14/mul_1:z:0"gru_14/while/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_14/while/gru_cell_14/add_3?
1gru_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_14_while_placeholder_1gru_14_while_placeholder"gru_14/while/gru_cell_14/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_14/while/TensorArrayV2Write/TensorListSetItemj
gru_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_14/while/add/y?
gru_14/while/addAddV2gru_14_while_placeholdergru_14/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_14/while/addn
gru_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_14/while/add_1/y?
gru_14/while/add_1AddV2&gru_14_while_gru_14_while_loop_countergru_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_14/while/add_1?
gru_14/while/IdentityIdentitygru_14/while/add_1:z:0^gru_14/while/NoOp*
T0*
_output_shapes
: 2
gru_14/while/Identity?
gru_14/while/Identity_1Identity,gru_14_while_gru_14_while_maximum_iterations^gru_14/while/NoOp*
T0*
_output_shapes
: 2
gru_14/while/Identity_1?
gru_14/while/Identity_2Identitygru_14/while/add:z:0^gru_14/while/NoOp*
T0*
_output_shapes
: 2
gru_14/while/Identity_2?
gru_14/while/Identity_3IdentityAgru_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_14/while/NoOp*
T0*
_output_shapes
: 2
gru_14/while/Identity_3?
gru_14/while/Identity_4Identity"gru_14/while/gru_cell_14/add_3:z:0^gru_14/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_14/while/Identity_4?
gru_14/while/NoOpNoOp/^gru_14/while/gru_cell_14/MatMul/ReadVariableOp1^gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp(^gru_14/while/gru_cell_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_14/while/NoOp"L
#gru_14_while_gru_14_strided_slice_1%gru_14_while_gru_14_strided_slice_1_0"x
9gru_14_while_gru_cell_14_matmul_1_readvariableop_resource;gru_14_while_gru_cell_14_matmul_1_readvariableop_resource_0"t
7gru_14_while_gru_cell_14_matmul_readvariableop_resource9gru_14_while_gru_cell_14_matmul_readvariableop_resource_0"f
0gru_14_while_gru_cell_14_readvariableop_resource2gru_14_while_gru_cell_14_readvariableop_resource_0"7
gru_14_while_identitygru_14/while/Identity:output:0";
gru_14_while_identity_1 gru_14/while/Identity_1:output:0";
gru_14_while_identity_2 gru_14/while/Identity_2:output:0";
gru_14_while_identity_3 gru_14/while/Identity_3:output:0";
gru_14_while_identity_4 gru_14/while/Identity_4:output:0"?
_gru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensoragru_14_while_tensorarrayv2read_tensorlistgetitem_gru_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_14/while/gru_cell_14/MatMul/ReadVariableOp.gru_14/while/gru_cell_14/MatMul/ReadVariableOp2d
0gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp0gru_14/while/gru_cell_14/MatMul_1/ReadVariableOp2R
'gru_14/while/gru_cell_14/ReadVariableOp'gru_14/while/gru_cell_14/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?!
?
H__inference_sequential_7_layer_call_and_return_conditional_losses_699031

inputs 
gru_14_698769:	? 
gru_14_698771:	?!
gru_14_698773:
?? 
gru_15_698936:	?!
gru_15_698938:
??!
gru_15_698940:
??#
dense_14_698982:
??
dense_14_698984:	?"
dense_15_699025:	?
dense_15_699027:
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?gru_14/StatefulPartitionedCall?gru_15/StatefulPartitionedCall?
gru_14/StatefulPartitionedCallStatefulPartitionedCallinputsgru_14_698769gru_14_698771gru_14_698773*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_14_layer_call_and_return_conditional_losses_6987682 
gru_14/StatefulPartitionedCall?
dropout_21/PartitionedCallPartitionedCall'gru_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_6987812
dropout_21/PartitionedCall?
gru_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0gru_15_698936gru_15_698938gru_15_698940*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_15_layer_call_and_return_conditional_losses_6989352 
gru_15/StatefulPartitionedCall?
dropout_22/PartitionedCallPartitionedCall'gru_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6989482
dropout_22/PartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_14_698982dense_14_698984*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_6989812"
 dense_14/StatefulPartitionedCall?
dropout_23/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_6989922
dropout_23/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_15_699025dense_15_699027*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_6990242"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall^gru_14/StatefulPartitionedCall^gru_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2@
gru_14/StatefulPartitionedCallgru_14/StatefulPartitionedCall2@
gru_15/StatefulPartitionedCallgru_15/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
,__inference_gru_cell_14_layer_call_fn_702045

inputs
states_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_6975472
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?&
?
H__inference_sequential_7_layer_call_and_return_conditional_losses_699661
gru_14_input 
gru_14_699633:	? 
gru_14_699635:	?!
gru_14_699637:
?? 
gru_15_699641:	?!
gru_15_699643:
??!
gru_15_699645:
??#
dense_14_699649:
??
dense_14_699651:	?"
dense_15_699655:	?
dense_15_699657:
identity?? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?"dropout_21/StatefulPartitionedCall?"dropout_22/StatefulPartitionedCall?"dropout_23/StatefulPartitionedCall?gru_14/StatefulPartitionedCall?gru_15/StatefulPartitionedCall?
gru_14/StatefulPartitionedCallStatefulPartitionedCallgru_14_inputgru_14_699633gru_14_699635gru_14_699637*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_14_layer_call_and_return_conditional_losses_6994842 
gru_14/StatefulPartitionedCall?
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall'gru_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_6993152$
"dropout_21/StatefulPartitionedCall?
gru_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0gru_15_699641gru_15_699643gru_15_699645*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_15_layer_call_and_return_conditional_losses_6992862 
gru_15/StatefulPartitionedCall?
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall'gru_15/StatefulPartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6991172$
"dropout_22/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_14_699649dense_14_699651*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_6989812"
 dense_14/StatefulPartitionedCall?
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_6990842$
"dropout_23/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_15_699655dense_15_699657*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_6990242"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall^gru_14/StatefulPartitionedCall^gru_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2@
gru_14/StatefulPartitionedCallgru_14/StatefulPartitionedCall2@
gru_15/StatefulPartitionedCallgru_15/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_14_input
?
?
while_cond_697559
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_697559___redundant_placeholder04
0while_while_cond_697559___redundant_placeholder14
0while_while_cond_697559___redundant_placeholder24
0while_while_cond_697559___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?E
?
while_body_701228
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_15_readvariableop_resource_0:	?F
2while_gru_cell_15_matmul_readvariableop_resource_0:
??H
4while_gru_cell_15_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_15_readvariableop_resource:	?D
0while_gru_cell_15_matmul_readvariableop_resource:
??F
2while_gru_cell_15_matmul_1_readvariableop_resource:
????'while/gru_cell_15/MatMul/ReadVariableOp?)while/gru_cell_15/MatMul_1/ReadVariableOp? while/gru_cell_15/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_15/ReadVariableOpReadVariableOp+while_gru_cell_15_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_15/ReadVariableOp?
while/gru_cell_15/unstackUnpack(while/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_15/unstack?
'while/gru_cell_15/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_15/MatMul/ReadVariableOp?
while/gru_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul?
while/gru_cell_15/BiasAddBiasAdd"while/gru_cell_15/MatMul:product:0"while/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd?
!while/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_15/split/split_dim?
while/gru_cell_15/splitSplit*while/gru_cell_15/split/split_dim:output:0"while/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split?
)while/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_15/MatMul_1/ReadVariableOp?
while/gru_cell_15/MatMul_1MatMulwhile_placeholder_21while/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul_1?
while/gru_cell_15/BiasAdd_1BiasAdd$while/gru_cell_15/MatMul_1:product:0"while/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd_1?
while/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_15/Const?
#while/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_15/split_1/split_dim?
while/gru_cell_15/split_1SplitV$while/gru_cell_15/BiasAdd_1:output:0 while/gru_cell_15/Const:output:0,while/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split_1?
while/gru_cell_15/addAddV2 while/gru_cell_15/split:output:0"while/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add?
while/gru_cell_15/SigmoidSigmoidwhile/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid?
while/gru_cell_15/add_1AddV2 while/gru_cell_15/split:output:1"while/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_1?
while/gru_cell_15/Sigmoid_1Sigmoidwhile/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid_1?
while/gru_cell_15/mulMulwhile/gru_cell_15/Sigmoid_1:y:0"while/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul?
while/gru_cell_15/add_2AddV2 while/gru_cell_15/split:output:2while/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_2?
while/gru_cell_15/ReluReluwhile/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Relu?
while/gru_cell_15/mul_1Mulwhile/gru_cell_15/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_1w
while/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_15/sub/x?
while/gru_cell_15/subSub while/gru_cell_15/sub/x:output:0while/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/sub?
while/gru_cell_15/mul_2Mulwhile/gru_cell_15/sub:z:0$while/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_2?
while/gru_cell_15/add_3AddV2while/gru_cell_15/mul_1:z:0while/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_15/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_15/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_15/MatMul/ReadVariableOp*^while/gru_cell_15/MatMul_1/ReadVariableOp!^while/gru_cell_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_15_matmul_1_readvariableop_resource4while_gru_cell_15_matmul_1_readvariableop_resource_0"f
0while_gru_cell_15_matmul_readvariableop_resource2while_gru_cell_15_matmul_readvariableop_resource_0"X
)while_gru_cell_15_readvariableop_resource+while_gru_cell_15_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_15/MatMul/ReadVariableOp'while/gru_cell_15/MatMul/ReadVariableOp2V
)while/gru_cell_15/MatMul_1/ReadVariableOp)while/gru_cell_15/MatMul_1/ReadVariableOp2D
 while/gru_cell_15/ReadVariableOp while/gru_cell_15/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_699196
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_699196___redundant_placeholder04
0while_while_cond_699196___redundant_placeholder14
0while_while_cond_699196___redundant_placeholder24
0while_while_cond_699196___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
gru_15_while_cond_700272*
&gru_15_while_gru_15_while_loop_counter0
,gru_15_while_gru_15_while_maximum_iterations
gru_15_while_placeholder
gru_15_while_placeholder_1
gru_15_while_placeholder_2,
(gru_15_while_less_gru_15_strided_slice_1B
>gru_15_while_gru_15_while_cond_700272___redundant_placeholder0B
>gru_15_while_gru_15_while_cond_700272___redundant_placeholder1B
>gru_15_while_gru_15_while_cond_700272___redundant_placeholder2B
>gru_15_while_gru_15_while_cond_700272___redundant_placeholder3
gru_15_while_identity
?
gru_15/while/LessLessgru_15_while_placeholder(gru_15_while_less_gru_15_strided_slice_1*
T0*
_output_shapes
: 2
gru_15/while/Lessr
gru_15/while/IdentityIdentitygru_15/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_15/while/Identity"7
gru_15_while_identitygru_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
d
F__inference_dropout_21_layer_call_and_return_conditional_losses_701142

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_gru_14_layer_call_fn_701115
inputs_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_14_layer_call_and_return_conditional_losses_6978172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
? 
?
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_702137

inputs
states_0*
readvariableop_resource:	?2
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:??????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?E
?
while_body_698679
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_14_readvariableop_resource_0:	?E
2while_gru_cell_14_matmul_readvariableop_resource_0:	?H
4while_gru_cell_14_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_14_readvariableop_resource:	?C
0while_gru_cell_14_matmul_readvariableop_resource:	?F
2while_gru_cell_14_matmul_1_readvariableop_resource:
????'while/gru_cell_14/MatMul/ReadVariableOp?)while/gru_cell_14/MatMul_1/ReadVariableOp? while/gru_cell_14/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_14/ReadVariableOpReadVariableOp+while_gru_cell_14_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_14/ReadVariableOp?
while/gru_cell_14/unstackUnpack(while/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_14/unstack?
'while/gru_cell_14/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_14/MatMul/ReadVariableOp?
while/gru_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul?
while/gru_cell_14/BiasAddBiasAdd"while/gru_cell_14/MatMul:product:0"while/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd?
!while/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_14/split/split_dim?
while/gru_cell_14/splitSplit*while/gru_cell_14/split/split_dim:output:0"while/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split?
)while/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_14/MatMul_1/ReadVariableOp?
while/gru_cell_14/MatMul_1MatMulwhile_placeholder_21while/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul_1?
while/gru_cell_14/BiasAdd_1BiasAdd$while/gru_cell_14/MatMul_1:product:0"while/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd_1?
while/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_14/Const?
#while/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_14/split_1/split_dim?
while/gru_cell_14/split_1SplitV$while/gru_cell_14/BiasAdd_1:output:0 while/gru_cell_14/Const:output:0,while/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split_1?
while/gru_cell_14/addAddV2 while/gru_cell_14/split:output:0"while/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add?
while/gru_cell_14/SigmoidSigmoidwhile/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid?
while/gru_cell_14/add_1AddV2 while/gru_cell_14/split:output:1"while/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_1?
while/gru_cell_14/Sigmoid_1Sigmoidwhile/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid_1?
while/gru_cell_14/mulMulwhile/gru_cell_14/Sigmoid_1:y:0"while/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul?
while/gru_cell_14/add_2AddV2 while/gru_cell_14/split:output:2while/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_2?
while/gru_cell_14/ReluReluwhile/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Relu?
while/gru_cell_14/mul_1Mulwhile/gru_cell_14/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_1w
while/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_14/sub/x?
while/gru_cell_14/subSub while/gru_cell_14/sub/x:output:0while/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/sub?
while/gru_cell_14/mul_2Mulwhile/gru_cell_14/sub:z:0$while/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_2?
while/gru_cell_14/add_3AddV2while/gru_cell_14/mul_1:z:0while/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_14/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_14/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_14/MatMul/ReadVariableOp*^while/gru_cell_14/MatMul_1/ReadVariableOp!^while/gru_cell_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_14_matmul_1_readvariableop_resource4while_gru_cell_14_matmul_1_readvariableop_resource_0"f
0while_gru_cell_14_matmul_readvariableop_resource2while_gru_cell_14_matmul_readvariableop_resource_0"X
)while_gru_cell_14_readvariableop_resource+while_gru_cell_14_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_14/MatMul/ReadVariableOp'while/gru_cell_14/MatMul/ReadVariableOp2V
)while/gru_cell_14/MatMul_1/ReadVariableOp)while/gru_cell_14/MatMul_1/ReadVariableOp2D
 while/gru_cell_14/ReadVariableOp while/gru_cell_14/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_21_layer_call_and_return_conditional_losses_699315

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_701227
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_701227___redundant_placeholder04
0while_while_cond_701227___redundant_placeholder14
0while_while_cond_701227___redundant_placeholder24
0while_while_cond_701227___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?Y
?
B__inference_gru_14_layer_call_and_return_conditional_losses_700634
inputs_06
#gru_cell_14_readvariableop_resource:	?=
*gru_cell_14_matmul_readvariableop_resource:	?@
,gru_cell_14_matmul_1_readvariableop_resource:
??
identity??!gru_cell_14/MatMul/ReadVariableOp?#gru_cell_14/MatMul_1/ReadVariableOp?gru_cell_14/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_14/ReadVariableOpReadVariableOp#gru_cell_14_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_14/ReadVariableOp?
gru_cell_14/unstackUnpack"gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_14/unstack?
!gru_cell_14/MatMul/ReadVariableOpReadVariableOp*gru_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_14/MatMul/ReadVariableOp?
gru_cell_14/MatMulMatMulstrided_slice_2:output:0)gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul?
gru_cell_14/BiasAddBiasAddgru_cell_14/MatMul:product:0gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd?
gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split/split_dim?
gru_cell_14/splitSplit$gru_cell_14/split/split_dim:output:0gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split?
#gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_14/MatMul_1/ReadVariableOp?
gru_cell_14/MatMul_1MatMulzeros:output:0+gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul_1?
gru_cell_14/BiasAdd_1BiasAddgru_cell_14/MatMul_1:product:0gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd_1{
gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_14/Const?
gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split_1/split_dim?
gru_cell_14/split_1SplitVgru_cell_14/BiasAdd_1:output:0gru_cell_14/Const:output:0&gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split_1?
gru_cell_14/addAddV2gru_cell_14/split:output:0gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add}
gru_cell_14/SigmoidSigmoidgru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid?
gru_cell_14/add_1AddV2gru_cell_14/split:output:1gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_1?
gru_cell_14/Sigmoid_1Sigmoidgru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid_1?
gru_cell_14/mulMulgru_cell_14/Sigmoid_1:y:0gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul?
gru_cell_14/add_2AddV2gru_cell_14/split:output:2gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_2v
gru_cell_14/ReluRelugru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Relu?
gru_cell_14/mul_1Mulgru_cell_14/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_1k
gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_14/sub/x?
gru_cell_14/subSubgru_cell_14/sub/x:output:0gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/sub?
gru_cell_14/mul_2Mulgru_cell_14/sub:z:0gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_2?
gru_cell_14/add_3AddV2gru_cell_14/mul_1:z:0gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_14_readvariableop_resource*gru_cell_14_matmul_readvariableop_resource,gru_cell_14_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_700545*
condR
while_cond_700544*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp"^gru_cell_14/MatMul/ReadVariableOp$^gru_cell_14/MatMul_1/ReadVariableOp^gru_cell_14/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2F
!gru_cell_14/MatMul/ReadVariableOp!gru_cell_14/MatMul/ReadVariableOp2J
#gru_cell_14/MatMul_1/ReadVariableOp#gru_cell_14/MatMul_1/ReadVariableOp28
gru_cell_14/ReadVariableOpgru_cell_14/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?E
?
while_body_699395
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_14_readvariableop_resource_0:	?E
2while_gru_cell_14_matmul_readvariableop_resource_0:	?H
4while_gru_cell_14_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_14_readvariableop_resource:	?C
0while_gru_cell_14_matmul_readvariableop_resource:	?F
2while_gru_cell_14_matmul_1_readvariableop_resource:
????'while/gru_cell_14/MatMul/ReadVariableOp?)while/gru_cell_14/MatMul_1/ReadVariableOp? while/gru_cell_14/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_14/ReadVariableOpReadVariableOp+while_gru_cell_14_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_14/ReadVariableOp?
while/gru_cell_14/unstackUnpack(while/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_14/unstack?
'while/gru_cell_14/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_14/MatMul/ReadVariableOp?
while/gru_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul?
while/gru_cell_14/BiasAddBiasAdd"while/gru_cell_14/MatMul:product:0"while/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd?
!while/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_14/split/split_dim?
while/gru_cell_14/splitSplit*while/gru_cell_14/split/split_dim:output:0"while/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split?
)while/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_14/MatMul_1/ReadVariableOp?
while/gru_cell_14/MatMul_1MatMulwhile_placeholder_21while/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul_1?
while/gru_cell_14/BiasAdd_1BiasAdd$while/gru_cell_14/MatMul_1:product:0"while/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd_1?
while/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_14/Const?
#while/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_14/split_1/split_dim?
while/gru_cell_14/split_1SplitV$while/gru_cell_14/BiasAdd_1:output:0 while/gru_cell_14/Const:output:0,while/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split_1?
while/gru_cell_14/addAddV2 while/gru_cell_14/split:output:0"while/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add?
while/gru_cell_14/SigmoidSigmoidwhile/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid?
while/gru_cell_14/add_1AddV2 while/gru_cell_14/split:output:1"while/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_1?
while/gru_cell_14/Sigmoid_1Sigmoidwhile/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid_1?
while/gru_cell_14/mulMulwhile/gru_cell_14/Sigmoid_1:y:0"while/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul?
while/gru_cell_14/add_2AddV2 while/gru_cell_14/split:output:2while/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_2?
while/gru_cell_14/ReluReluwhile/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Relu?
while/gru_cell_14/mul_1Mulwhile/gru_cell_14/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_1w
while/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_14/sub/x?
while/gru_cell_14/subSub while/gru_cell_14/sub/x:output:0while/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/sub?
while/gru_cell_14/mul_2Mulwhile/gru_cell_14/sub:z:0$while/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_2?
while/gru_cell_14/add_3AddV2while/gru_cell_14/mul_1:z:0while/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_14/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_14/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_14/MatMul/ReadVariableOp*^while/gru_cell_14/MatMul_1/ReadVariableOp!^while/gru_cell_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_14_matmul_1_readvariableop_resource4while_gru_cell_14_matmul_1_readvariableop_resource_0"f
0while_gru_cell_14_matmul_readvariableop_resource2while_gru_cell_14_matmul_readvariableop_resource_0"X
)while_gru_cell_14_readvariableop_resource+while_gru_cell_14_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_14/MatMul/ReadVariableOp'while/gru_cell_14/MatMul/ReadVariableOp2V
)while/gru_cell_14/MatMul_1/ReadVariableOp)while/gru_cell_14/MatMul_1/ReadVariableOp2D
 while/gru_cell_14/ReadVariableOp while/gru_cell_14/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_23_layer_call_and_return_conditional_losses_701904

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?X
?
B__inference_gru_15_layer_call_and_return_conditional_losses_699286

inputs6
#gru_cell_15_readvariableop_resource:	?>
*gru_cell_15_matmul_readvariableop_resource:
??@
,gru_cell_15_matmul_1_readvariableop_resource:
??
identity??!gru_cell_15/MatMul/ReadVariableOp?#gru_cell_15/MatMul_1/ReadVariableOp?gru_cell_15/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_15/ReadVariableOpReadVariableOp#gru_cell_15_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_15/ReadVariableOp?
gru_cell_15/unstackUnpack"gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_15/unstack?
!gru_cell_15/MatMul/ReadVariableOpReadVariableOp*gru_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_15/MatMul/ReadVariableOp?
gru_cell_15/MatMulMatMulstrided_slice_2:output:0)gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul?
gru_cell_15/BiasAddBiasAddgru_cell_15/MatMul:product:0gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd?
gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split/split_dim?
gru_cell_15/splitSplit$gru_cell_15/split/split_dim:output:0gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split?
#gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_15/MatMul_1/ReadVariableOp?
gru_cell_15/MatMul_1MatMulzeros:output:0+gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul_1?
gru_cell_15/BiasAdd_1BiasAddgru_cell_15/MatMul_1:product:0gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd_1{
gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_15/Const?
gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split_1/split_dim?
gru_cell_15/split_1SplitVgru_cell_15/BiasAdd_1:output:0gru_cell_15/Const:output:0&gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split_1?
gru_cell_15/addAddV2gru_cell_15/split:output:0gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add}
gru_cell_15/SigmoidSigmoidgru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid?
gru_cell_15/add_1AddV2gru_cell_15/split:output:1gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_1?
gru_cell_15/Sigmoid_1Sigmoidgru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid_1?
gru_cell_15/mulMulgru_cell_15/Sigmoid_1:y:0gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul?
gru_cell_15/add_2AddV2gru_cell_15/split:output:2gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_2v
gru_cell_15/ReluRelugru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Relu?
gru_cell_15/mul_1Mulgru_cell_15/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_1k
gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_15/sub/x?
gru_cell_15/subSubgru_cell_15/sub/x:output:0gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/sub?
gru_cell_15/mul_2Mulgru_cell_15/sub:z:0gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_2?
gru_cell_15/add_3AddV2gru_cell_15/mul_1:z:0gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_15_readvariableop_resource*gru_cell_15_matmul_readvariableop_resource,gru_cell_15_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_699197*
condR
while_cond_699196*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp"^gru_cell_15/MatMul/ReadVariableOp$^gru_cell_15/MatMul_1/ReadVariableOp^gru_cell_15/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_15/MatMul/ReadVariableOp!gru_cell_15/MatMul/ReadVariableOp2J
#gru_cell_15/MatMul_1/ReadVariableOp#gru_cell_15/MatMul_1/ReadVariableOp28
gru_cell_15/ReadVariableOpgru_cell_15/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?E
?
while_body_698846
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_15_readvariableop_resource_0:	?F
2while_gru_cell_15_matmul_readvariableop_resource_0:
??H
4while_gru_cell_15_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_15_readvariableop_resource:	?D
0while_gru_cell_15_matmul_readvariableop_resource:
??F
2while_gru_cell_15_matmul_1_readvariableop_resource:
????'while/gru_cell_15/MatMul/ReadVariableOp?)while/gru_cell_15/MatMul_1/ReadVariableOp? while/gru_cell_15/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_15/ReadVariableOpReadVariableOp+while_gru_cell_15_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_15/ReadVariableOp?
while/gru_cell_15/unstackUnpack(while/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_15/unstack?
'while/gru_cell_15/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_15/MatMul/ReadVariableOp?
while/gru_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul?
while/gru_cell_15/BiasAddBiasAdd"while/gru_cell_15/MatMul:product:0"while/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd?
!while/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_15/split/split_dim?
while/gru_cell_15/splitSplit*while/gru_cell_15/split/split_dim:output:0"while/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split?
)while/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_15/MatMul_1/ReadVariableOp?
while/gru_cell_15/MatMul_1MatMulwhile_placeholder_21while/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul_1?
while/gru_cell_15/BiasAdd_1BiasAdd$while/gru_cell_15/MatMul_1:product:0"while/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd_1?
while/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_15/Const?
#while/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_15/split_1/split_dim?
while/gru_cell_15/split_1SplitV$while/gru_cell_15/BiasAdd_1:output:0 while/gru_cell_15/Const:output:0,while/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split_1?
while/gru_cell_15/addAddV2 while/gru_cell_15/split:output:0"while/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add?
while/gru_cell_15/SigmoidSigmoidwhile/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid?
while/gru_cell_15/add_1AddV2 while/gru_cell_15/split:output:1"while/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_1?
while/gru_cell_15/Sigmoid_1Sigmoidwhile/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid_1?
while/gru_cell_15/mulMulwhile/gru_cell_15/Sigmoid_1:y:0"while/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul?
while/gru_cell_15/add_2AddV2 while/gru_cell_15/split:output:2while/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_2?
while/gru_cell_15/ReluReluwhile/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Relu?
while/gru_cell_15/mul_1Mulwhile/gru_cell_15/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_1w
while/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_15/sub/x?
while/gru_cell_15/subSub while/gru_cell_15/sub/x:output:0while/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/sub?
while/gru_cell_15/mul_2Mulwhile/gru_cell_15/sub:z:0$while/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_2?
while/gru_cell_15/add_3AddV2while/gru_cell_15/mul_1:z:0while/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_15/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_15/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_15/MatMul/ReadVariableOp*^while/gru_cell_15/MatMul_1/ReadVariableOp!^while/gru_cell_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_15_matmul_1_readvariableop_resource4while_gru_cell_15_matmul_1_readvariableop_resource_0"f
0while_gru_cell_15_matmul_readvariableop_resource2while_gru_cell_15_matmul_readvariableop_resource_0"X
)while_gru_cell_15_readvariableop_resource+while_gru_cell_15_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_15/MatMul/ReadVariableOp'while/gru_cell_15/MatMul/ReadVariableOp2V
)while/gru_cell_15/MatMul_1/ReadVariableOp)while/gru_cell_15/MatMul_1/ReadVariableOp2D
 while/gru_cell_15/ReadVariableOp while/gru_cell_15/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?!
?
D__inference_dense_14_layer_call_and_return_conditional_losses_701878

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_22_layer_call_and_return_conditional_losses_699117

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?E
?
while_body_701381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_15_readvariableop_resource_0:	?F
2while_gru_cell_15_matmul_readvariableop_resource_0:
??H
4while_gru_cell_15_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_15_readvariableop_resource:	?D
0while_gru_cell_15_matmul_readvariableop_resource:
??F
2while_gru_cell_15_matmul_1_readvariableop_resource:
????'while/gru_cell_15/MatMul/ReadVariableOp?)while/gru_cell_15/MatMul_1/ReadVariableOp? while/gru_cell_15/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_15/ReadVariableOpReadVariableOp+while_gru_cell_15_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_15/ReadVariableOp?
while/gru_cell_15/unstackUnpack(while/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_15/unstack?
'while/gru_cell_15/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_15/MatMul/ReadVariableOp?
while/gru_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul?
while/gru_cell_15/BiasAddBiasAdd"while/gru_cell_15/MatMul:product:0"while/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd?
!while/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_15/split/split_dim?
while/gru_cell_15/splitSplit*while/gru_cell_15/split/split_dim:output:0"while/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split?
)while/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_15/MatMul_1/ReadVariableOp?
while/gru_cell_15/MatMul_1MatMulwhile_placeholder_21while/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul_1?
while/gru_cell_15/BiasAdd_1BiasAdd$while/gru_cell_15/MatMul_1:product:0"while/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd_1?
while/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_15/Const?
#while/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_15/split_1/split_dim?
while/gru_cell_15/split_1SplitV$while/gru_cell_15/BiasAdd_1:output:0 while/gru_cell_15/Const:output:0,while/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split_1?
while/gru_cell_15/addAddV2 while/gru_cell_15/split:output:0"while/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add?
while/gru_cell_15/SigmoidSigmoidwhile/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid?
while/gru_cell_15/add_1AddV2 while/gru_cell_15/split:output:1"while/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_1?
while/gru_cell_15/Sigmoid_1Sigmoidwhile/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid_1?
while/gru_cell_15/mulMulwhile/gru_cell_15/Sigmoid_1:y:0"while/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul?
while/gru_cell_15/add_2AddV2 while/gru_cell_15/split:output:2while/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_2?
while/gru_cell_15/ReluReluwhile/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Relu?
while/gru_cell_15/mul_1Mulwhile/gru_cell_15/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_1w
while/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_15/sub/x?
while/gru_cell_15/subSub while/gru_cell_15/sub/x:output:0while/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/sub?
while/gru_cell_15/mul_2Mulwhile/gru_cell_15/sub:z:0$while/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_2?
while/gru_cell_15/add_3AddV2while/gru_cell_15/mul_1:z:0while/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_15/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_15/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_15/MatMul/ReadVariableOp*^while/gru_cell_15/MatMul_1/ReadVariableOp!^while/gru_cell_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_15_matmul_1_readvariableop_resource4while_gru_cell_15_matmul_1_readvariableop_resource_0"f
0while_gru_cell_15_matmul_readvariableop_resource2while_gru_cell_15_matmul_readvariableop_resource_0"X
)while_gru_cell_15_readvariableop_resource+while_gru_cell_15_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_15/MatMul/ReadVariableOp'while/gru_cell_15/MatMul/ReadVariableOp2V
)while/gru_cell_15/MatMul_1/ReadVariableOp)while/gru_cell_15/MatMul_1/ReadVariableOp2D
 while/gru_cell_15/ReadVariableOp while/gru_cell_15/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_700850
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_700850___redundant_placeholder04
0while_while_cond_700850___redundant_placeholder14
0while_while_cond_700850___redundant_placeholder24
0while_while_cond_700850___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?"
?
while_body_697560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_14_697582_0:	?-
while_gru_cell_14_697584_0:	?.
while_gru_cell_14_697586_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_14_697582:	?+
while_gru_cell_14_697584:	?,
while_gru_cell_14_697586:
????)while/gru_cell_14/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_14_697582_0while_gru_cell_14_697584_0while_gru_cell_14_697586_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_6975472+
)while/gru_cell_14/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_14/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/gru_cell_14/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"6
while_gru_cell_14_697582while_gru_cell_14_697582_0"6
while_gru_cell_14_697584while_gru_cell_14_697584_0"6
while_gru_cell_14_697586while_gru_cell_14_697586_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2V
)while/gru_cell_14/StatefulPartitionedCall)while/gru_cell_14/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_gru_15_layer_call_fn_701787
inputs_0
unknown:	?
	unknown_0:
??
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_15_layer_call_and_return_conditional_losses_6981902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?P
?	
gru_15_while_body_700273*
&gru_15_while_gru_15_while_loop_counter0
,gru_15_while_gru_15_while_maximum_iterations
gru_15_while_placeholder
gru_15_while_placeholder_1
gru_15_while_placeholder_2)
%gru_15_while_gru_15_strided_slice_1_0e
agru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0E
2gru_15_while_gru_cell_15_readvariableop_resource_0:	?M
9gru_15_while_gru_cell_15_matmul_readvariableop_resource_0:
??O
;gru_15_while_gru_cell_15_matmul_1_readvariableop_resource_0:
??
gru_15_while_identity
gru_15_while_identity_1
gru_15_while_identity_2
gru_15_while_identity_3
gru_15_while_identity_4'
#gru_15_while_gru_15_strided_slice_1c
_gru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensorC
0gru_15_while_gru_cell_15_readvariableop_resource:	?K
7gru_15_while_gru_cell_15_matmul_readvariableop_resource:
??M
9gru_15_while_gru_cell_15_matmul_1_readvariableop_resource:
????.gru_15/while/gru_cell_15/MatMul/ReadVariableOp?0gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp?'gru_15/while/gru_cell_15/ReadVariableOp?
>gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_15/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0gru_15_while_placeholderGgru_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0gru_15/while/TensorArrayV2Read/TensorListGetItem?
'gru_15/while/gru_cell_15/ReadVariableOpReadVariableOp2gru_15_while_gru_cell_15_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_15/while/gru_cell_15/ReadVariableOp?
 gru_15/while/gru_cell_15/unstackUnpack/gru_15/while/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_15/while/gru_cell_15/unstack?
.gru_15/while/gru_cell_15/MatMul/ReadVariableOpReadVariableOp9gru_15_while_gru_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype020
.gru_15/while/gru_cell_15/MatMul/ReadVariableOp?
gru_15/while/gru_cell_15/MatMulMatMul7gru_15/while/TensorArrayV2Read/TensorListGetItem:item:06gru_15/while/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_15/while/gru_cell_15/MatMul?
 gru_15/while/gru_cell_15/BiasAddBiasAdd)gru_15/while/gru_cell_15/MatMul:product:0)gru_15/while/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_15/while/gru_cell_15/BiasAdd?
(gru_15/while/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_15/while/gru_cell_15/split/split_dim?
gru_15/while/gru_cell_15/splitSplit1gru_15/while/gru_cell_15/split/split_dim:output:0)gru_15/while/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_15/while/gru_cell_15/split?
0gru_15/while/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp;gru_15_while_gru_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp?
!gru_15/while/gru_cell_15/MatMul_1MatMulgru_15_while_placeholder_28gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_15/while/gru_cell_15/MatMul_1?
"gru_15/while/gru_cell_15/BiasAdd_1BiasAdd+gru_15/while/gru_cell_15/MatMul_1:product:0)gru_15/while/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_15/while/gru_cell_15/BiasAdd_1?
gru_15/while/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2 
gru_15/while/gru_cell_15/Const?
*gru_15/while/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_15/while/gru_cell_15/split_1/split_dim?
 gru_15/while/gru_cell_15/split_1SplitV+gru_15/while/gru_cell_15/BiasAdd_1:output:0'gru_15/while/gru_cell_15/Const:output:03gru_15/while/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_15/while/gru_cell_15/split_1?
gru_15/while/gru_cell_15/addAddV2'gru_15/while/gru_cell_15/split:output:0)gru_15/while/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_15/while/gru_cell_15/add?
 gru_15/while/gru_cell_15/SigmoidSigmoid gru_15/while/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_15/while/gru_cell_15/Sigmoid?
gru_15/while/gru_cell_15/add_1AddV2'gru_15/while/gru_cell_15/split:output:1)gru_15/while/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_15/while/gru_cell_15/add_1?
"gru_15/while/gru_cell_15/Sigmoid_1Sigmoid"gru_15/while/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_15/while/gru_cell_15/Sigmoid_1?
gru_15/while/gru_cell_15/mulMul&gru_15/while/gru_cell_15/Sigmoid_1:y:0)gru_15/while/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_15/while/gru_cell_15/mul?
gru_15/while/gru_cell_15/add_2AddV2'gru_15/while/gru_cell_15/split:output:2 gru_15/while/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_15/while/gru_cell_15/add_2?
gru_15/while/gru_cell_15/ReluRelu"gru_15/while/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_15/while/gru_cell_15/Relu?
gru_15/while/gru_cell_15/mul_1Mul$gru_15/while/gru_cell_15/Sigmoid:y:0gru_15_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_15/while/gru_cell_15/mul_1?
gru_15/while/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_15/while/gru_cell_15/sub/x?
gru_15/while/gru_cell_15/subSub'gru_15/while/gru_cell_15/sub/x:output:0$gru_15/while/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_15/while/gru_cell_15/sub?
gru_15/while/gru_cell_15/mul_2Mul gru_15/while/gru_cell_15/sub:z:0+gru_15/while/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_15/while/gru_cell_15/mul_2?
gru_15/while/gru_cell_15/add_3AddV2"gru_15/while/gru_cell_15/mul_1:z:0"gru_15/while/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_15/while/gru_cell_15/add_3?
1gru_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_15_while_placeholder_1gru_15_while_placeholder"gru_15/while/gru_cell_15/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_15/while/TensorArrayV2Write/TensorListSetItemj
gru_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_15/while/add/y?
gru_15/while/addAddV2gru_15_while_placeholdergru_15/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_15/while/addn
gru_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_15/while/add_1/y?
gru_15/while/add_1AddV2&gru_15_while_gru_15_while_loop_countergru_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_15/while/add_1?
gru_15/while/IdentityIdentitygru_15/while/add_1:z:0^gru_15/while/NoOp*
T0*
_output_shapes
: 2
gru_15/while/Identity?
gru_15/while/Identity_1Identity,gru_15_while_gru_15_while_maximum_iterations^gru_15/while/NoOp*
T0*
_output_shapes
: 2
gru_15/while/Identity_1?
gru_15/while/Identity_2Identitygru_15/while/add:z:0^gru_15/while/NoOp*
T0*
_output_shapes
: 2
gru_15/while/Identity_2?
gru_15/while/Identity_3IdentityAgru_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_15/while/NoOp*
T0*
_output_shapes
: 2
gru_15/while/Identity_3?
gru_15/while/Identity_4Identity"gru_15/while/gru_cell_15/add_3:z:0^gru_15/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_15/while/Identity_4?
gru_15/while/NoOpNoOp/^gru_15/while/gru_cell_15/MatMul/ReadVariableOp1^gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp(^gru_15/while/gru_cell_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_15/while/NoOp"L
#gru_15_while_gru_15_strided_slice_1%gru_15_while_gru_15_strided_slice_1_0"x
9gru_15_while_gru_cell_15_matmul_1_readvariableop_resource;gru_15_while_gru_cell_15_matmul_1_readvariableop_resource_0"t
7gru_15_while_gru_cell_15_matmul_readvariableop_resource9gru_15_while_gru_cell_15_matmul_readvariableop_resource_0"f
0gru_15_while_gru_cell_15_readvariableop_resource2gru_15_while_gru_cell_15_readvariableop_resource_0"7
gru_15_while_identitygru_15/while/Identity:output:0";
gru_15_while_identity_1 gru_15/while/Identity_1:output:0";
gru_15_while_identity_2 gru_15/while/Identity_2:output:0";
gru_15_while_identity_3 gru_15/while/Identity_3:output:0";
gru_15_while_identity_4 gru_15/while/Identity_4:output:0"?
_gru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensoragru_15_while_tensorarrayv2read_tensorlistgetitem_gru_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_15/while/gru_cell_15/MatMul/ReadVariableOp.gru_15/while/gru_cell_15/MatMul/ReadVariableOp2d
0gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp0gru_15/while/gru_cell_15/MatMul_1/ReadVariableOp2R
'gru_15/while/gru_cell_15/ReadVariableOp'gru_15/while/gru_cell_15/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?"
?
while_body_698319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_15_698341_0:	?.
while_gru_cell_15_698343_0:
??.
while_gru_cell_15_698345_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_15_698341:	?,
while_gru_cell_15_698343:
??,
while_gru_cell_15_698345:
????)while/gru_cell_15/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_15/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_15_698341_0while_gru_cell_15_698343_0while_gru_cell_15_698345_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_6982562+
)while/gru_cell_15/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_15/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/gru_cell_15/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"6
while_gru_cell_15_698341while_gru_cell_15_698341_0"6
while_gru_cell_15_698343while_gru_cell_15_698343_0"6
while_gru_cell_15_698345while_gru_cell_15_698345_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2V
)while/gru_cell_15/StatefulPartitionedCall)while/gru_cell_15/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?"
?
while_body_698126
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_15_698148_0:	?.
while_gru_cell_15_698150_0:
??.
while_gru_cell_15_698152_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_15_698148:	?,
while_gru_cell_15_698150:
??,
while_gru_cell_15_698152:
????)while/gru_cell_15/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_15/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_15_698148_0while_gru_cell_15_698150_0while_gru_cell_15_698152_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_6981132+
)while/gru_cell_15/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_15/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/gru_cell_15/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"6
while_gru_cell_15_698148while_gru_cell_15_698148_0"6
while_gru_cell_15_698150while_gru_cell_15_698150_0"6
while_gru_cell_15_698152while_gru_cell_15_698152_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2V
)while/gru_cell_15/StatefulPartitionedCall)while/gru_cell_15/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
? 
?
D__inference_dense_15_layer_call_and_return_conditional_losses_699024

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?Y
?
B__inference_gru_15_layer_call_and_return_conditional_losses_701317
inputs_06
#gru_cell_15_readvariableop_resource:	?>
*gru_cell_15_matmul_readvariableop_resource:
??@
,gru_cell_15_matmul_1_readvariableop_resource:
??
identity??!gru_cell_15/MatMul/ReadVariableOp?#gru_cell_15/MatMul_1/ReadVariableOp?gru_cell_15/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_15/ReadVariableOpReadVariableOp#gru_cell_15_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_15/ReadVariableOp?
gru_cell_15/unstackUnpack"gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_15/unstack?
!gru_cell_15/MatMul/ReadVariableOpReadVariableOp*gru_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_15/MatMul/ReadVariableOp?
gru_cell_15/MatMulMatMulstrided_slice_2:output:0)gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul?
gru_cell_15/BiasAddBiasAddgru_cell_15/MatMul:product:0gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd?
gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split/split_dim?
gru_cell_15/splitSplit$gru_cell_15/split/split_dim:output:0gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split?
#gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_15/MatMul_1/ReadVariableOp?
gru_cell_15/MatMul_1MatMulzeros:output:0+gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul_1?
gru_cell_15/BiasAdd_1BiasAddgru_cell_15/MatMul_1:product:0gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd_1{
gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_15/Const?
gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split_1/split_dim?
gru_cell_15/split_1SplitVgru_cell_15/BiasAdd_1:output:0gru_cell_15/Const:output:0&gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split_1?
gru_cell_15/addAddV2gru_cell_15/split:output:0gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add}
gru_cell_15/SigmoidSigmoidgru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid?
gru_cell_15/add_1AddV2gru_cell_15/split:output:1gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_1?
gru_cell_15/Sigmoid_1Sigmoidgru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid_1?
gru_cell_15/mulMulgru_cell_15/Sigmoid_1:y:0gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul?
gru_cell_15/add_2AddV2gru_cell_15/split:output:2gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_2v
gru_cell_15/ReluRelugru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Relu?
gru_cell_15/mul_1Mulgru_cell_15/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_1k
gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_15/sub/x?
gru_cell_15/subSubgru_cell_15/sub/x:output:0gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/sub?
gru_cell_15/mul_2Mulgru_cell_15/sub:z:0gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_2?
gru_cell_15/add_3AddV2gru_cell_15/mul_1:z:0gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_15_readvariableop_resource*gru_cell_15_matmul_readvariableop_resource,gru_cell_15_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_701228*
condR
while_cond_701227*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp"^gru_cell_15/MatMul/ReadVariableOp$^gru_cell_15/MatMul_1/ReadVariableOp^gru_cell_15/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2F
!gru_cell_15/MatMul/ReadVariableOp!gru_cell_15/MatMul/ReadVariableOp2J
#gru_cell_15/MatMul_1/ReadVariableOp#gru_cell_15/MatMul_1/ReadVariableOp28
gru_cell_15/ReadVariableOpgru_cell_15/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?E
?
while_body_700698
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_14_readvariableop_resource_0:	?E
2while_gru_cell_14_matmul_readvariableop_resource_0:	?H
4while_gru_cell_14_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_14_readvariableop_resource:	?C
0while_gru_cell_14_matmul_readvariableop_resource:	?F
2while_gru_cell_14_matmul_1_readvariableop_resource:
????'while/gru_cell_14/MatMul/ReadVariableOp?)while/gru_cell_14/MatMul_1/ReadVariableOp? while/gru_cell_14/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_14/ReadVariableOpReadVariableOp+while_gru_cell_14_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_14/ReadVariableOp?
while/gru_cell_14/unstackUnpack(while/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_14/unstack?
'while/gru_cell_14/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_14/MatMul/ReadVariableOp?
while/gru_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul?
while/gru_cell_14/BiasAddBiasAdd"while/gru_cell_14/MatMul:product:0"while/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd?
!while/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_14/split/split_dim?
while/gru_cell_14/splitSplit*while/gru_cell_14/split/split_dim:output:0"while/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split?
)while/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_14/MatMul_1/ReadVariableOp?
while/gru_cell_14/MatMul_1MatMulwhile_placeholder_21while/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul_1?
while/gru_cell_14/BiasAdd_1BiasAdd$while/gru_cell_14/MatMul_1:product:0"while/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd_1?
while/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_14/Const?
#while/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_14/split_1/split_dim?
while/gru_cell_14/split_1SplitV$while/gru_cell_14/BiasAdd_1:output:0 while/gru_cell_14/Const:output:0,while/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split_1?
while/gru_cell_14/addAddV2 while/gru_cell_14/split:output:0"while/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add?
while/gru_cell_14/SigmoidSigmoidwhile/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid?
while/gru_cell_14/add_1AddV2 while/gru_cell_14/split:output:1"while/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_1?
while/gru_cell_14/Sigmoid_1Sigmoidwhile/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid_1?
while/gru_cell_14/mulMulwhile/gru_cell_14/Sigmoid_1:y:0"while/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul?
while/gru_cell_14/add_2AddV2 while/gru_cell_14/split:output:2while/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_2?
while/gru_cell_14/ReluReluwhile/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Relu?
while/gru_cell_14/mul_1Mulwhile/gru_cell_14/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_1w
while/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_14/sub/x?
while/gru_cell_14/subSub while/gru_cell_14/sub/x:output:0while/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/sub?
while/gru_cell_14/mul_2Mulwhile/gru_cell_14/sub:z:0$while/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_2?
while/gru_cell_14/add_3AddV2while/gru_cell_14/mul_1:z:0while/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_14/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_14/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_14/MatMul/ReadVariableOp*^while/gru_cell_14/MatMul_1/ReadVariableOp!^while/gru_cell_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_14_matmul_1_readvariableop_resource4while_gru_cell_14_matmul_1_readvariableop_resource_0"f
0while_gru_cell_14_matmul_readvariableop_resource2while_gru_cell_14_matmul_readvariableop_resource_0"X
)while_gru_cell_14_readvariableop_resource+while_gru_cell_14_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_14/MatMul/ReadVariableOp'while/gru_cell_14/MatMul/ReadVariableOp2V
)while/gru_cell_14/MatMul_1/ReadVariableOp)while/gru_cell_14/MatMul_1/ReadVariableOp2D
 while/gru_cell_14/ReadVariableOp while/gru_cell_14/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
,__inference_gru_cell_14_layer_call_fn_702059

inputs
states_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_6976902
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:?????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_701892

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
D__inference_dense_15_layer_call_and_return_conditional_losses_701944

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_gru_15_layer_call_fn_701798
inputs_0
unknown:	?
	unknown_0:
??
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_gru_15_layer_call_and_return_conditional_losses_6983832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?E
?
while_body_700851
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_14_readvariableop_resource_0:	?E
2while_gru_cell_14_matmul_readvariableop_resource_0:	?H
4while_gru_cell_14_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_14_readvariableop_resource:	?C
0while_gru_cell_14_matmul_readvariableop_resource:	?F
2while_gru_cell_14_matmul_1_readvariableop_resource:
????'while/gru_cell_14/MatMul/ReadVariableOp?)while/gru_cell_14/MatMul_1/ReadVariableOp? while/gru_cell_14/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_14/ReadVariableOpReadVariableOp+while_gru_cell_14_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_14/ReadVariableOp?
while/gru_cell_14/unstackUnpack(while/gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_14/unstack?
'while/gru_cell_14/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_14/MatMul/ReadVariableOp?
while/gru_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul?
while/gru_cell_14/BiasAddBiasAdd"while/gru_cell_14/MatMul:product:0"while/gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd?
!while/gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_14/split/split_dim?
while/gru_cell_14/splitSplit*while/gru_cell_14/split/split_dim:output:0"while/gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split?
)while/gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_14/MatMul_1/ReadVariableOp?
while/gru_cell_14/MatMul_1MatMulwhile_placeholder_21while/gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/MatMul_1?
while/gru_cell_14/BiasAdd_1BiasAdd$while/gru_cell_14/MatMul_1:product:0"while/gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/BiasAdd_1?
while/gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_14/Const?
#while/gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_14/split_1/split_dim?
while/gru_cell_14/split_1SplitV$while/gru_cell_14/BiasAdd_1:output:0 while/gru_cell_14/Const:output:0,while/gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_14/split_1?
while/gru_cell_14/addAddV2 while/gru_cell_14/split:output:0"while/gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add?
while/gru_cell_14/SigmoidSigmoidwhile/gru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid?
while/gru_cell_14/add_1AddV2 while/gru_cell_14/split:output:1"while/gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_1?
while/gru_cell_14/Sigmoid_1Sigmoidwhile/gru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Sigmoid_1?
while/gru_cell_14/mulMulwhile/gru_cell_14/Sigmoid_1:y:0"while/gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul?
while/gru_cell_14/add_2AddV2 while/gru_cell_14/split:output:2while/gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_2?
while/gru_cell_14/ReluReluwhile/gru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/Relu?
while/gru_cell_14/mul_1Mulwhile/gru_cell_14/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_1w
while/gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_14/sub/x?
while/gru_cell_14/subSub while/gru_cell_14/sub/x:output:0while/gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/sub?
while/gru_cell_14/mul_2Mulwhile/gru_cell_14/sub:z:0$while/gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/mul_2?
while/gru_cell_14/add_3AddV2while/gru_cell_14/mul_1:z:0while/gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_14/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_14/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_14/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_14/MatMul/ReadVariableOp*^while/gru_cell_14/MatMul_1/ReadVariableOp!^while/gru_cell_14/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_14_matmul_1_readvariableop_resource4while_gru_cell_14_matmul_1_readvariableop_resource_0"f
0while_gru_cell_14_matmul_readvariableop_resource2while_gru_cell_14_matmul_readvariableop_resource_0"X
)while_gru_cell_14_readvariableop_resource+while_gru_cell_14_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_14/MatMul/ReadVariableOp'while/gru_cell_14/MatMul/ReadVariableOp2V
)while/gru_cell_14/MatMul_1/ReadVariableOp)while/gru_cell_14/MatMul_1/ReadVariableOp2D
 while/gru_cell_14/ReadVariableOp while/gru_cell_14/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_23_layer_call_and_return_conditional_losses_699084

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
while_body_697753
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_14_697775_0:	?-
while_gru_cell_14_697777_0:	?.
while_gru_cell_14_697779_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_14_697775:	?+
while_gru_cell_14_697777:	?,
while_gru_cell_14_697779:
????)while/gru_cell_14/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_14_697775_0while_gru_cell_14_697777_0while_gru_cell_14_697779_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_6976902+
)while/gru_cell_14/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_14/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/gru_cell_14/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"6
while_gru_cell_14_697775while_gru_cell_14_697775_0"6
while_gru_cell_14_697777while_gru_cell_14_697777_0"6
while_gru_cell_14_697779while_gru_cell_14_697779_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2V
)while/gru_cell_14/StatefulPartitionedCall)while/gru_cell_14/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?X
?
B__inference_gru_15_layer_call_and_return_conditional_losses_698935

inputs6
#gru_cell_15_readvariableop_resource:	?>
*gru_cell_15_matmul_readvariableop_resource:
??@
,gru_cell_15_matmul_1_readvariableop_resource:
??
identity??!gru_cell_15/MatMul/ReadVariableOp?#gru_cell_15/MatMul_1/ReadVariableOp?gru_cell_15/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_15/ReadVariableOpReadVariableOp#gru_cell_15_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_15/ReadVariableOp?
gru_cell_15/unstackUnpack"gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_15/unstack?
!gru_cell_15/MatMul/ReadVariableOpReadVariableOp*gru_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_15/MatMul/ReadVariableOp?
gru_cell_15/MatMulMatMulstrided_slice_2:output:0)gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul?
gru_cell_15/BiasAddBiasAddgru_cell_15/MatMul:product:0gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd?
gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split/split_dim?
gru_cell_15/splitSplit$gru_cell_15/split/split_dim:output:0gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split?
#gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_15/MatMul_1/ReadVariableOp?
gru_cell_15/MatMul_1MatMulzeros:output:0+gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/MatMul_1?
gru_cell_15/BiasAdd_1BiasAddgru_cell_15/MatMul_1:product:0gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/BiasAdd_1{
gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_15/Const?
gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_15/split_1/split_dim?
gru_cell_15/split_1SplitVgru_cell_15/BiasAdd_1:output:0gru_cell_15/Const:output:0&gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_15/split_1?
gru_cell_15/addAddV2gru_cell_15/split:output:0gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add}
gru_cell_15/SigmoidSigmoidgru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid?
gru_cell_15/add_1AddV2gru_cell_15/split:output:1gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_1?
gru_cell_15/Sigmoid_1Sigmoidgru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Sigmoid_1?
gru_cell_15/mulMulgru_cell_15/Sigmoid_1:y:0gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul?
gru_cell_15/add_2AddV2gru_cell_15/split:output:2gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_2v
gru_cell_15/ReluRelugru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/Relu?
gru_cell_15/mul_1Mulgru_cell_15/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_1k
gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_15/sub/x?
gru_cell_15/subSubgru_cell_15/sub/x:output:0gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/sub?
gru_cell_15/mul_2Mulgru_cell_15/sub:z:0gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/mul_2?
gru_cell_15/add_3AddV2gru_cell_15/mul_1:z:0gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_15/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_15_readvariableop_resource*gru_cell_15_matmul_readvariableop_resource,gru_cell_15_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_698846*
condR
while_cond_698845*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp"^gru_cell_15/MatMul/ReadVariableOp$^gru_cell_15/MatMul_1/ReadVariableOp^gru_cell_15/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_15/MatMul/ReadVariableOp!gru_cell_15/MatMul/ReadVariableOp2J
#gru_cell_15/MatMul_1/ReadVariableOp#gru_cell_15/MatMul_1/ReadVariableOp28
gru_cell_15/ReadVariableOpgru_cell_15/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?E
?
while_body_701687
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_15_readvariableop_resource_0:	?F
2while_gru_cell_15_matmul_readvariableop_resource_0:
??H
4while_gru_cell_15_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_15_readvariableop_resource:	?D
0while_gru_cell_15_matmul_readvariableop_resource:
??F
2while_gru_cell_15_matmul_1_readvariableop_resource:
????'while/gru_cell_15/MatMul/ReadVariableOp?)while/gru_cell_15/MatMul_1/ReadVariableOp? while/gru_cell_15/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_15/ReadVariableOpReadVariableOp+while_gru_cell_15_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_15/ReadVariableOp?
while/gru_cell_15/unstackUnpack(while/gru_cell_15/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_15/unstack?
'while/gru_cell_15/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_15/MatMul/ReadVariableOp?
while/gru_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul?
while/gru_cell_15/BiasAddBiasAdd"while/gru_cell_15/MatMul:product:0"while/gru_cell_15/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd?
!while/gru_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_15/split/split_dim?
while/gru_cell_15/splitSplit*while/gru_cell_15/split/split_dim:output:0"while/gru_cell_15/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split?
)while/gru_cell_15/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_15/MatMul_1/ReadVariableOp?
while/gru_cell_15/MatMul_1MatMulwhile_placeholder_21while/gru_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/MatMul_1?
while/gru_cell_15/BiasAdd_1BiasAdd$while/gru_cell_15/MatMul_1:product:0"while/gru_cell_15/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/BiasAdd_1?
while/gru_cell_15/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_15/Const?
#while/gru_cell_15/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_15/split_1/split_dim?
while/gru_cell_15/split_1SplitV$while/gru_cell_15/BiasAdd_1:output:0 while/gru_cell_15/Const:output:0,while/gru_cell_15/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_15/split_1?
while/gru_cell_15/addAddV2 while/gru_cell_15/split:output:0"while/gru_cell_15/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add?
while/gru_cell_15/SigmoidSigmoidwhile/gru_cell_15/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid?
while/gru_cell_15/add_1AddV2 while/gru_cell_15/split:output:1"while/gru_cell_15/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_1?
while/gru_cell_15/Sigmoid_1Sigmoidwhile/gru_cell_15/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Sigmoid_1?
while/gru_cell_15/mulMulwhile/gru_cell_15/Sigmoid_1:y:0"while/gru_cell_15/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul?
while/gru_cell_15/add_2AddV2 while/gru_cell_15/split:output:2while/gru_cell_15/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_2?
while/gru_cell_15/ReluReluwhile/gru_cell_15/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/Relu?
while/gru_cell_15/mul_1Mulwhile/gru_cell_15/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_1w
while/gru_cell_15/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_15/sub/x?
while/gru_cell_15/subSub while/gru_cell_15/sub/x:output:0while/gru_cell_15/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/sub?
while/gru_cell_15/mul_2Mulwhile/gru_cell_15/sub:z:0$while/gru_cell_15/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/mul_2?
while/gru_cell_15/add_3AddV2while/gru_cell_15/mul_1:z:0while/gru_cell_15/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_15/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_15/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell_15/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_15/MatMul/ReadVariableOp*^while/gru_cell_15/MatMul_1/ReadVariableOp!^while/gru_cell_15/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_15_matmul_1_readvariableop_resource4while_gru_cell_15_matmul_1_readvariableop_resource_0"f
0while_gru_cell_15_matmul_readvariableop_resource2while_gru_cell_15_matmul_readvariableop_resource_0"X
)while_gru_cell_15_readvariableop_resource+while_gru_cell_15_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_15/MatMul/ReadVariableOp'while/gru_cell_15/MatMul/ReadVariableOp2V
)while/gru_cell_15/MatMul_1/ReadVariableOp)while/gru_cell_15/MatMul_1/ReadVariableOp2D
 while/gru_cell_15/ReadVariableOp while/gru_cell_15/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_698113

inputs

states*
readvariableop_resource:	?2
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:??????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?X
?
B__inference_gru_14_layer_call_and_return_conditional_losses_700940

inputs6
#gru_cell_14_readvariableop_resource:	?=
*gru_cell_14_matmul_readvariableop_resource:	?@
,gru_cell_14_matmul_1_readvariableop_resource:
??
identity??!gru_cell_14/MatMul/ReadVariableOp?#gru_cell_14/MatMul_1/ReadVariableOp?gru_cell_14/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_14/ReadVariableOpReadVariableOp#gru_cell_14_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_14/ReadVariableOp?
gru_cell_14/unstackUnpack"gru_cell_14/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_14/unstack?
!gru_cell_14/MatMul/ReadVariableOpReadVariableOp*gru_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_14/MatMul/ReadVariableOp?
gru_cell_14/MatMulMatMulstrided_slice_2:output:0)gru_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul?
gru_cell_14/BiasAddBiasAddgru_cell_14/MatMul:product:0gru_cell_14/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd?
gru_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split/split_dim?
gru_cell_14/splitSplit$gru_cell_14/split/split_dim:output:0gru_cell_14/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split?
#gru_cell_14/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_14/MatMul_1/ReadVariableOp?
gru_cell_14/MatMul_1MatMulzeros:output:0+gru_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/MatMul_1?
gru_cell_14/BiasAdd_1BiasAddgru_cell_14/MatMul_1:product:0gru_cell_14/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/BiasAdd_1{
gru_cell_14/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_14/Const?
gru_cell_14/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_14/split_1/split_dim?
gru_cell_14/split_1SplitVgru_cell_14/BiasAdd_1:output:0gru_cell_14/Const:output:0&gru_cell_14/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_14/split_1?
gru_cell_14/addAddV2gru_cell_14/split:output:0gru_cell_14/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add}
gru_cell_14/SigmoidSigmoidgru_cell_14/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid?
gru_cell_14/add_1AddV2gru_cell_14/split:output:1gru_cell_14/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_1?
gru_cell_14/Sigmoid_1Sigmoidgru_cell_14/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Sigmoid_1?
gru_cell_14/mulMulgru_cell_14/Sigmoid_1:y:0gru_cell_14/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul?
gru_cell_14/add_2AddV2gru_cell_14/split:output:2gru_cell_14/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_2v
gru_cell_14/ReluRelugru_cell_14/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/Relu?
gru_cell_14/mul_1Mulgru_cell_14/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_1k
gru_cell_14/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_14/sub/x?
gru_cell_14/subSubgru_cell_14/sub/x:output:0gru_cell_14/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/sub?
gru_cell_14/mul_2Mulgru_cell_14/sub:z:0gru_cell_14/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/mul_2?
gru_cell_14/add_3AddV2gru_cell_14/mul_1:z:0gru_cell_14/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_14/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_14_readvariableop_resource*gru_cell_14_matmul_readvariableop_resource,gru_cell_14_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_700851*
condR
while_cond_700850*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp"^gru_cell_14/MatMul/ReadVariableOp$^gru_cell_14/MatMul_1/ReadVariableOp^gru_cell_14/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_14/MatMul/ReadVariableOp!gru_cell_14/MatMul/ReadVariableOp2J
#gru_cell_14/MatMul_1/ReadVariableOp#gru_cell_14/MatMul_1/ReadVariableOp28
gru_cell_14/ReadVariableOpgru_cell_14/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
+__inference_dropout_22_layer_call_fn_701847

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6991172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_23_layer_call_fn_701909

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_6989922
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_702098

inputs
states_0*
readvariableop_resource:	?2
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
Constq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:??????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?;
?
B__inference_gru_14_layer_call_and_return_conditional_losses_697817

inputs%
gru_cell_14_697741:	?%
gru_cell_14_697743:	?&
gru_cell_14_697745:
??
identity??#gru_cell_14/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicec
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_14_697741gru_cell_14_697743gru_cell_14_697745*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_6976902%
#gru_cell_14/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_14_697741gru_cell_14_697743gru_cell_14_697745*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_697753*
condR
while_cond_697752*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity|
NoOpNoOp$^gru_cell_14/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#gru_cell_14/StatefulPartitionedCall#gru_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
gru_14_input9
serving_default_gru_14_input:0?????????@
dense_154
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_sequential
?
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_rnn_layer
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_rnn_layer
?
trainable_variables
regularization_losses
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
2iter

3beta_1

4beta_2
	5decay
6learning_rate"m?#m?,m?-m?7m?8m?9m?:m?;m?<m?"v?#v?,v?-v?7v?8v?9v?:v?;v?<v?"
	optimizer
f
70
81
92
:3
;4
<5
"6
#7
,8
-9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
70
81
92
:3
;4
<5
"6
#7
,8
-9"
trackable_list_wrapper
?
	trainable_variables

regularization_losses
=layer_regularization_losses

>layers
?layer_metrics
	variables
@non_trainable_variables
Ametrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

7kernel
8recurrent_kernel
9bias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
?
trainable_variables

Fstates
regularization_losses
Glayer_regularization_losses

Hlayers
Ilayer_metrics
	variables
Jnon_trainable_variables
Kmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
regularization_losses
	variables
Llayer_regularization_losses

Mlayers
Nlayer_metrics
Onon_trainable_variables
Pmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

:kernel
;recurrent_kernel
<bias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
?
trainable_variables

Ustates
regularization_losses
Vlayer_regularization_losses

Wlayers
Xlayer_metrics
	variables
Ynon_trainable_variables
Zmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
regularization_losses
 	variables
[layer_regularization_losses

\layers
]layer_metrics
^non_trainable_variables
_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_14/kernel
:?2dense_14/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
$trainable_variables
%regularization_losses
&	variables
`layer_regularization_losses

alayers
blayer_metrics
cnon_trainable_variables
dmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
(trainable_variables
)regularization_losses
*	variables
elayer_regularization_losses

flayers
glayer_metrics
hnon_trainable_variables
imetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_15/kernel
:2dense_15/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
.trainable_variables
/regularization_losses
0	variables
jlayer_regularization_losses

klayers
llayer_metrics
mnon_trainable_variables
nmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	?2gru_14/gru_cell_14/kernel
7:5
??2#gru_14/gru_cell_14/recurrent_kernel
*:(	?2gru_14/gru_cell_14/bias
-:+
??2gru_15/gru_cell_15/kernel
7:5
??2#gru_15/gru_cell_15/recurrent_kernel
*:(	?2gru_15/gru_cell_15/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
 "
trackable_list_wrapper
5
70
81
92"
trackable_list_wrapper
?
Btrainable_variables
Cregularization_losses
D	variables
qlayer_regularization_losses

rlayers
slayer_metrics
tnon_trainable_variables
umetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
?
Qtrainable_variables
Rregularization_losses
S	variables
vlayer_regularization_losses

wlayers
xlayer_metrics
ynon_trainable_variables
zmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	{total
	|count
}	variables
~	keras_api"
_tf_keras_metric
b
	total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
{0
|1"
trackable_list_wrapper
-
}	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/
0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
(:&
??2Adam/dense_14/kernel/m
!:?2Adam/dense_14/bias/m
':%	?2Adam/dense_15/kernel/m
 :2Adam/dense_15/bias/m
1:/	?2 Adam/gru_14/gru_cell_14/kernel/m
<::
??2*Adam/gru_14/gru_cell_14/recurrent_kernel/m
/:-	?2Adam/gru_14/gru_cell_14/bias/m
2:0
??2 Adam/gru_15/gru_cell_15/kernel/m
<::
??2*Adam/gru_15/gru_cell_15/recurrent_kernel/m
/:-	?2Adam/gru_15/gru_cell_15/bias/m
(:&
??2Adam/dense_14/kernel/v
!:?2Adam/dense_14/bias/v
':%	?2Adam/dense_15/kernel/v
 :2Adam/dense_15/bias/v
1:/	?2 Adam/gru_14/gru_cell_14/kernel/v
<::
??2*Adam/gru_14/gru_cell_14/recurrent_kernel/v
/:-	?2Adam/gru_14/gru_cell_14/bias/v
2:0
??2 Adam/gru_15/gru_cell_15/kernel/v
<::
??2*Adam/gru_15/gru_cell_15/recurrent_kernel/v
/:-	?2Adam/gru_15/gru_cell_15/bias/v
?2?
H__inference_sequential_7_layer_call_and_return_conditional_losses_700052
H__inference_sequential_7_layer_call_and_return_conditional_losses_700431
H__inference_sequential_7_layer_call_and_return_conditional_losses_699630
H__inference_sequential_7_layer_call_and_return_conditional_losses_699661?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_697477gru_14_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_sequential_7_layer_call_fn_699054
-__inference_sequential_7_layer_call_fn_700456
-__inference_sequential_7_layer_call_fn_700481
-__inference_sequential_7_layer_call_fn_699599?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_gru_14_layer_call_and_return_conditional_losses_700634
B__inference_gru_14_layer_call_and_return_conditional_losses_700787
B__inference_gru_14_layer_call_and_return_conditional_losses_700940
B__inference_gru_14_layer_call_and_return_conditional_losses_701093?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_gru_14_layer_call_fn_701104
'__inference_gru_14_layer_call_fn_701115
'__inference_gru_14_layer_call_fn_701126
'__inference_gru_14_layer_call_fn_701137?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_21_layer_call_and_return_conditional_losses_701142
F__inference_dropout_21_layer_call_and_return_conditional_losses_701154?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_21_layer_call_fn_701159
+__inference_dropout_21_layer_call_fn_701164?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_gru_15_layer_call_and_return_conditional_losses_701317
B__inference_gru_15_layer_call_and_return_conditional_losses_701470
B__inference_gru_15_layer_call_and_return_conditional_losses_701623
B__inference_gru_15_layer_call_and_return_conditional_losses_701776?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_gru_15_layer_call_fn_701787
'__inference_gru_15_layer_call_fn_701798
'__inference_gru_15_layer_call_fn_701809
'__inference_gru_15_layer_call_fn_701820?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_22_layer_call_and_return_conditional_losses_701825
F__inference_dropout_22_layer_call_and_return_conditional_losses_701837?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_22_layer_call_fn_701842
+__inference_dropout_22_layer_call_fn_701847?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dense_14_layer_call_and_return_conditional_losses_701878?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_14_layer_call_fn_701887?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout_23_layer_call_and_return_conditional_losses_701892
F__inference_dropout_23_layer_call_and_return_conditional_losses_701904?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_23_layer_call_fn_701909
+__inference_dropout_23_layer_call_fn_701914?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dense_15_layer_call_and_return_conditional_losses_701944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_15_layer_call_fn_701953?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_699694gru_14_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_701992
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_702031?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_gru_cell_14_layer_call_fn_702045
,__inference_gru_cell_14_layer_call_fn_702059?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_702098
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_702137?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_gru_cell_15_layer_call_fn_702151
,__inference_gru_cell_15_layer_call_fn_702165?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
!__inference__wrapped_model_697477?
978<:;"#,-9?6
/?,
*?'
gru_14_input?????????
? "7?4
2
dense_15&?#
dense_15??????????
D__inference_dense_14_layer_call_and_return_conditional_losses_701878f"#4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
)__inference_dense_14_layer_call_fn_701887Y"#4?1
*?'
%?"
inputs??????????
? "????????????
D__inference_dense_15_layer_call_and_return_conditional_losses_701944e,-4?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
)__inference_dense_15_layer_call_fn_701953X,-4?1
*?'
%?"
inputs??????????
? "???????????
F__inference_dropout_21_layer_call_and_return_conditional_losses_701142f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
F__inference_dropout_21_layer_call_and_return_conditional_losses_701154f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
+__inference_dropout_21_layer_call_fn_701159Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
+__inference_dropout_21_layer_call_fn_701164Y8?5
.?+
%?"
inputs??????????
p
? "????????????
F__inference_dropout_22_layer_call_and_return_conditional_losses_701825f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
F__inference_dropout_22_layer_call_and_return_conditional_losses_701837f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
+__inference_dropout_22_layer_call_fn_701842Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
+__inference_dropout_22_layer_call_fn_701847Y8?5
.?+
%?"
inputs??????????
p
? "????????????
F__inference_dropout_23_layer_call_and_return_conditional_losses_701892f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
F__inference_dropout_23_layer_call_and_return_conditional_losses_701904f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
+__inference_dropout_23_layer_call_fn_701909Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
+__inference_dropout_23_layer_call_fn_701914Y8?5
.?+
%?"
inputs??????????
p
? "????????????
B__inference_gru_14_layer_call_and_return_conditional_losses_700634?978O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
B__inference_gru_14_layer_call_and_return_conditional_losses_700787?978O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
B__inference_gru_14_layer_call_and_return_conditional_losses_700940r978??<
5?2
$?!
inputs?????????

 
p 

 
? "*?'
 ?
0??????????
? ?
B__inference_gru_14_layer_call_and_return_conditional_losses_701093r978??<
5?2
$?!
inputs?????????

 
p

 
? "*?'
 ?
0??????????
? ?
'__inference_gru_14_layer_call_fn_701104~978O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "&?#????????????????????
'__inference_gru_14_layer_call_fn_701115~978O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "&?#????????????????????
'__inference_gru_14_layer_call_fn_701126e978??<
5?2
$?!
inputs?????????

 
p 

 
? "????????????
'__inference_gru_14_layer_call_fn_701137e978??<
5?2
$?!
inputs?????????

 
p

 
? "????????????
B__inference_gru_15_layer_call_and_return_conditional_losses_701317?<:;P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
B__inference_gru_15_layer_call_and_return_conditional_losses_701470?<:;P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
B__inference_gru_15_layer_call_and_return_conditional_losses_701623s<:;@?=
6?3
%?"
inputs??????????

 
p 

 
? "*?'
 ?
0??????????
? ?
B__inference_gru_15_layer_call_and_return_conditional_losses_701776s<:;@?=
6?3
%?"
inputs??????????

 
p

 
? "*?'
 ?
0??????????
? ?
'__inference_gru_15_layer_call_fn_701787<:;P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#????????????????????
'__inference_gru_15_layer_call_fn_701798<:;P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#????????????????????
'__inference_gru_15_layer_call_fn_701809f<:;@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
'__inference_gru_15_layer_call_fn_701820f<:;@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_701992?978]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p 
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
G__inference_gru_cell_14_layer_call_and_return_conditional_losses_702031?978]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
,__inference_gru_cell_14_layer_call_fn_702045?978]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p 
? "F?C
?
0??????????
#? 
?
1/0???????????
,__inference_gru_cell_14_layer_call_fn_702059?978]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p
? "F?C
?
0??????????
#? 
?
1/0???????????
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_702098?<:;^?[
T?Q
!?
inputs??????????
(?%
#? 
states/0??????????
p 
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
G__inference_gru_cell_15_layer_call_and_return_conditional_losses_702137?<:;^?[
T?Q
!?
inputs??????????
(?%
#? 
states/0??????????
p
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
,__inference_gru_cell_15_layer_call_fn_702151?<:;^?[
T?Q
!?
inputs??????????
(?%
#? 
states/0??????????
p 
? "F?C
?
0??????????
#? 
?
1/0???????????
,__inference_gru_cell_15_layer_call_fn_702165?<:;^?[
T?Q
!?
inputs??????????
(?%
#? 
states/0??????????
p
? "F?C
?
0??????????
#? 
?
1/0???????????
H__inference_sequential_7_layer_call_and_return_conditional_losses_699630z
978<:;"#,-A?>
7?4
*?'
gru_14_input?????????
p 

 
? ")?&
?
0?????????
? ?
H__inference_sequential_7_layer_call_and_return_conditional_losses_699661z
978<:;"#,-A?>
7?4
*?'
gru_14_input?????????
p

 
? ")?&
?
0?????????
? ?
H__inference_sequential_7_layer_call_and_return_conditional_losses_700052t
978<:;"#,-;?8
1?.
$?!
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
H__inference_sequential_7_layer_call_and_return_conditional_losses_700431t
978<:;"#,-;?8
1?.
$?!
inputs?????????
p

 
? ")?&
?
0?????????
? ?
-__inference_sequential_7_layer_call_fn_699054m
978<:;"#,-A?>
7?4
*?'
gru_14_input?????????
p 

 
? "???????????
-__inference_sequential_7_layer_call_fn_699599m
978<:;"#,-A?>
7?4
*?'
gru_14_input?????????
p

 
? "???????????
-__inference_sequential_7_layer_call_fn_700456g
978<:;"#,-;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
-__inference_sequential_7_layer_call_fn_700481g
978<:;"#,-;?8
1?.
$?!
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_699694?
978<:;"#,-I?F
? 
??<
:
gru_14_input*?'
gru_14_input?????????"7?4
2
dense_15&?#
dense_15?????????