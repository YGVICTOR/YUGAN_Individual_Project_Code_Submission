??-
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
?"serve*2.6.02unknown8??+
|
dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_63/kernel
u
#dense_63/kernel/Read/ReadVariableOpReadVariableOpdense_63/kernel* 
_output_shapes
:
??*
dtype0
s
dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_63/bias
l
!dense_63/bias/Read/ReadVariableOpReadVariableOpdense_63/bias*
_output_shapes	
:?*
dtype0
|
dense_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_64/kernel
u
#dense_64/kernel/Read/ReadVariableOpReadVariableOpdense_64/kernel* 
_output_shapes
:
??*
dtype0
s
dense_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_64/bias
l
!dense_64/bias/Read/ReadVariableOpReadVariableOpdense_64/bias*
_output_shapes	
:?*
dtype0
{
dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_65/kernel
t
#dense_65/kernel/Read/ReadVariableOpReadVariableOpdense_65/kernel*
_output_shapes
:	?*
dtype0
r
dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_65/bias
k
!dense_65/bias/Read/ReadVariableOpReadVariableOpdense_65/bias*
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
gru_56/gru_cell_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namegru_56/gru_cell_56/kernel
?
-gru_56/gru_cell_56/kernel/Read/ReadVariableOpReadVariableOpgru_56/gru_cell_56/kernel*
_output_shapes
:	?*
dtype0
?
#gru_56/gru_cell_56/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#gru_56/gru_cell_56/recurrent_kernel
?
7gru_56/gru_cell_56/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_56/gru_cell_56/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
gru_56/gru_cell_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_namegru_56/gru_cell_56/bias
?
+gru_56/gru_cell_56/bias/Read/ReadVariableOpReadVariableOpgru_56/gru_cell_56/bias*
_output_shapes
:	?*
dtype0
?
gru_57/gru_cell_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_namegru_57/gru_cell_57/kernel
?
-gru_57/gru_cell_57/kernel/Read/ReadVariableOpReadVariableOpgru_57/gru_cell_57/kernel* 
_output_shapes
:
??*
dtype0
?
#gru_57/gru_cell_57/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#gru_57/gru_cell_57/recurrent_kernel
?
7gru_57/gru_cell_57/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_57/gru_cell_57/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
gru_57/gru_cell_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_namegru_57/gru_cell_57/bias
?
+gru_57/gru_cell_57/bias/Read/ReadVariableOpReadVariableOpgru_57/gru_cell_57/bias*
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
Adam/dense_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_63/kernel/m
?
*Adam/dense_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_63/bias/m
z
(Adam/dense_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_64/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_64/kernel/m
?
*Adam/dense_64/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_64/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_64/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_64/bias/m
z
(Adam/dense_64/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_64/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_65/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_65/kernel/m
?
*Adam/dense_65/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_65/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_65/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_65/bias/m
y
(Adam/dense_65/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_65/bias/m*
_output_shapes
:*
dtype0
?
 Adam/gru_56/gru_cell_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/gru_56/gru_cell_56/kernel/m
?
4Adam/gru_56/gru_cell_56/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_56/gru_cell_56/kernel/m*
_output_shapes
:	?*
dtype0
?
*Adam/gru_56/gru_cell_56/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_56/gru_cell_56/recurrent_kernel/m
?
>Adam/gru_56/gru_cell_56/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_56/gru_cell_56/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/gru_56/gru_cell_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_56/gru_cell_56/bias/m
?
2Adam/gru_56/gru_cell_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_56/gru_cell_56/bias/m*
_output_shapes
:	?*
dtype0
?
 Adam/gru_57/gru_cell_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/gru_57/gru_cell_57/kernel/m
?
4Adam/gru_57/gru_cell_57/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_57/gru_cell_57/kernel/m* 
_output_shapes
:
??*
dtype0
?
*Adam/gru_57/gru_cell_57/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_57/gru_cell_57/recurrent_kernel/m
?
>Adam/gru_57/gru_cell_57/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_57/gru_cell_57/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/gru_57/gru_cell_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_57/gru_cell_57/bias/m
?
2Adam/gru_57/gru_cell_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_57/gru_cell_57/bias/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_63/kernel/v
?
*Adam/dense_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_63/bias/v
z
(Adam/dense_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_64/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_64/kernel/v
?
*Adam/dense_64/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_64/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_64/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_64/bias/v
z
(Adam/dense_64/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_64/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_65/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_65/kernel/v
?
*Adam/dense_65/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_65/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_65/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_65/bias/v
y
(Adam/dense_65/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_65/bias/v*
_output_shapes
:*
dtype0
?
 Adam/gru_56/gru_cell_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/gru_56/gru_cell_56/kernel/v
?
4Adam/gru_56/gru_cell_56/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_56/gru_cell_56/kernel/v*
_output_shapes
:	?*
dtype0
?
*Adam/gru_56/gru_cell_56/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_56/gru_cell_56/recurrent_kernel/v
?
>Adam/gru_56/gru_cell_56/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_56/gru_cell_56/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/gru_56/gru_cell_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_56/gru_cell_56/bias/v
?
2Adam/gru_56/gru_cell_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_56/gru_cell_56/bias/v*
_output_shapes
:	?*
dtype0
?
 Adam/gru_57/gru_cell_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/gru_57/gru_cell_57/kernel/v
?
4Adam/gru_57/gru_cell_57/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_57/gru_cell_57/kernel/v* 
_output_shapes
:
??*
dtype0
?
*Adam/gru_57/gru_cell_57/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_57/gru_cell_57/recurrent_kernel/v
?
>Adam/gru_57/gru_cell_57/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_57/gru_cell_57/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/gru_57/gru_cell_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_57/gru_cell_57/bias/v
?
2Adam/gru_57/gru_cell_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_57/gru_cell_57/bias/v*
_output_shapes
:	?*
dtype0

NoOpNoOp
?L
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?K
value?KB?K B?K
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
layer-7
	layer_with_weights-4
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
h

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
R
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
R
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
?
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate$m?%m?.m?/m?8m?9m?Cm?Dm?Em?Fm?Gm?Hm?$v?%v?.v?/v?8v?9v?Cv?Dv?Ev?Fv?Gv?Hv?
V
C0
D1
E2
F3
G4
H5
$6
%7
.8
/9
810
911
V
C0
D1
E2
F3
G4
H5
$6
%7
.8
/9
810
911
 
?
	variables

Ilayers
Jlayer_regularization_losses
Kmetrics
trainable_variables
Lnon_trainable_variables
Mlayer_metrics
regularization_losses
 
~

Ckernel
Drecurrent_kernel
Ebias
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
 

C0
D1
E2

C0
D1
E2
 
?
	variables

Rlayers
Slayer_regularization_losses

Tstates
Umetrics
trainable_variables
Vnon_trainable_variables
Wlayer_metrics
regularization_losses
 
 
 
?
	variables
Xlayer_regularization_losses

Ylayers
regularization_losses
Zmetrics
trainable_variables
[layer_metrics
\non_trainable_variables
~

Fkernel
Grecurrent_kernel
Hbias
]	variables
^regularization_losses
_trainable_variables
`	keras_api
 

F0
G1
H2

F0
G1
H2
 
?
	variables

alayers
blayer_regularization_losses

cstates
dmetrics
trainable_variables
enon_trainable_variables
flayer_metrics
regularization_losses
 
 
 
?
 	variables
glayer_regularization_losses

hlayers
!regularization_losses
imetrics
"trainable_variables
jlayer_metrics
knon_trainable_variables
[Y
VARIABLE_VALUEdense_63/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_63/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
?
&	variables
llayer_regularization_losses

mlayers
'regularization_losses
nmetrics
(trainable_variables
olayer_metrics
pnon_trainable_variables
 
 
 
?
*	variables
qlayer_regularization_losses

rlayers
+regularization_losses
smetrics
,trainable_variables
tlayer_metrics
unon_trainable_variables
[Y
VARIABLE_VALUEdense_64/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_64/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
?
0	variables
vlayer_regularization_losses

wlayers
1regularization_losses
xmetrics
2trainable_variables
ylayer_metrics
znon_trainable_variables
 
 
 
?
4	variables
{layer_regularization_losses

|layers
5regularization_losses
}metrics
6trainable_variables
~layer_metrics
non_trainable_variables
[Y
VARIABLE_VALUEdense_65/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_65/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
?
:	variables
 ?layer_regularization_losses
?layers
;regularization_losses
?metrics
<trainable_variables
?layer_metrics
?non_trainable_variables
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
US
VARIABLE_VALUEgru_56/gru_cell_56/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#gru_56/gru_cell_56/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEgru_56/gru_cell_56/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEgru_57/gru_cell_57/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#gru_57/gru_cell_57/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEgru_57/gru_cell_57/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
?
0
1
2
3
4
5
6
7
	8
 

?0
?1
 
 

C0
D1
E2
 

C0
D1
E2
?
N	variables
 ?layer_regularization_losses
?layers
Oregularization_losses
?metrics
Ptrainable_variables
?layer_metrics
?non_trainable_variables

0
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

F0
G1
H2
 

F0
G1
H2
?
]	variables
 ?layer_regularization_losses
?layers
^regularization_losses
?metrics
_trainable_variables
?layer_metrics
?non_trainable_variables

0
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
8

?total

?count
?	variables
?	keras_api
I

?total

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

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/dense_63/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_63/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_64/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_64/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_65/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_65/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/gru_56/gru_cell_56/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_56/gru_cell_56/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_56/gru_cell_56/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/gru_57/gru_cell_57/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_57/gru_cell_57/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_57/gru_cell_57/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_63/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_63/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_64/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_64/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_65/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_65/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/gru_56/gru_cell_56/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_56/gru_cell_56/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_56/gru_cell_56/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/gru_57/gru_cell_57/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_57/gru_cell_57/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gru_57/gru_cell_57/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_gru_56_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_56_inputgru_56/gru_cell_56/biasgru_56/gru_cell_56/kernel#gru_56/gru_cell_56/recurrent_kernelgru_57/gru_cell_57/biasgru_57/gru_cell_57/kernel#gru_57/gru_cell_57/recurrent_kerneldense_63/kerneldense_63/biasdense_64/kerneldense_64/biasdense_65/kerneldense_65/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3599042
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_63/kernel/Read/ReadVariableOp!dense_63/bias/Read/ReadVariableOp#dense_64/kernel/Read/ReadVariableOp!dense_64/bias/Read/ReadVariableOp#dense_65/kernel/Read/ReadVariableOp!dense_65/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-gru_56/gru_cell_56/kernel/Read/ReadVariableOp7gru_56/gru_cell_56/recurrent_kernel/Read/ReadVariableOp+gru_56/gru_cell_56/bias/Read/ReadVariableOp-gru_57/gru_cell_57/kernel/Read/ReadVariableOp7gru_57/gru_cell_57/recurrent_kernel/Read/ReadVariableOp+gru_57/gru_cell_57/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_63/kernel/m/Read/ReadVariableOp(Adam/dense_63/bias/m/Read/ReadVariableOp*Adam/dense_64/kernel/m/Read/ReadVariableOp(Adam/dense_64/bias/m/Read/ReadVariableOp*Adam/dense_65/kernel/m/Read/ReadVariableOp(Adam/dense_65/bias/m/Read/ReadVariableOp4Adam/gru_56/gru_cell_56/kernel/m/Read/ReadVariableOp>Adam/gru_56/gru_cell_56/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_56/gru_cell_56/bias/m/Read/ReadVariableOp4Adam/gru_57/gru_cell_57/kernel/m/Read/ReadVariableOp>Adam/gru_57/gru_cell_57/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_57/gru_cell_57/bias/m/Read/ReadVariableOp*Adam/dense_63/kernel/v/Read/ReadVariableOp(Adam/dense_63/bias/v/Read/ReadVariableOp*Adam/dense_64/kernel/v/Read/ReadVariableOp(Adam/dense_64/bias/v/Read/ReadVariableOp*Adam/dense_65/kernel/v/Read/ReadVariableOp(Adam/dense_65/bias/v/Read/ReadVariableOp4Adam/gru_56/gru_cell_56/kernel/v/Read/ReadVariableOp>Adam/gru_56/gru_cell_56/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_56/gru_cell_56/bias/v/Read/ReadVariableOp4Adam/gru_57/gru_cell_57/kernel/v/Read/ReadVariableOp>Adam/gru_57/gru_cell_57/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_57/gru_cell_57/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_3601809
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_63/kerneldense_63/biasdense_64/kerneldense_64/biasdense_65/kerneldense_65/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_56/gru_cell_56/kernel#gru_56/gru_cell_56/recurrent_kernelgru_56/gru_cell_56/biasgru_57/gru_cell_57/kernel#gru_57/gru_cell_57/recurrent_kernelgru_57/gru_cell_57/biastotalcounttotal_1count_1Adam/dense_63/kernel/mAdam/dense_63/bias/mAdam/dense_64/kernel/mAdam/dense_64/bias/mAdam/dense_65/kernel/mAdam/dense_65/bias/m Adam/gru_56/gru_cell_56/kernel/m*Adam/gru_56/gru_cell_56/recurrent_kernel/mAdam/gru_56/gru_cell_56/bias/m Adam/gru_57/gru_cell_57/kernel/m*Adam/gru_57/gru_cell_57/recurrent_kernel/mAdam/gru_57/gru_cell_57/bias/mAdam/dense_63/kernel/vAdam/dense_63/bias/vAdam/dense_64/kernel/vAdam/dense_64/bias/vAdam/dense_65/kernel/vAdam/dense_65/bias/v Adam/gru_56/gru_cell_56/kernel/v*Adam/gru_56/gru_cell_56/recurrent_kernel/vAdam/gru_56/gru_cell_56/bias/v Adam/gru_57/gru_cell_57/kernel/v*Adam/gru_57/gru_cell_57/recurrent_kernel/vAdam/gru_57/gru_cell_57/bias/v*9
Tin2
02.*
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_3601954??)
?
?
*__inference_dense_65_layer_call_fn_3601409

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
GPU 2J 8? *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_35983012
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
?E
?
while_body_3598511
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_57_readvariableop_resource_0:	?F
2while_gru_cell_57_matmul_readvariableop_resource_0:
??H
4while_gru_cell_57_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_57_readvariableop_resource:	?D
0while_gru_cell_57_matmul_readvariableop_resource:
??F
2while_gru_cell_57_matmul_1_readvariableop_resource:
????'while/gru_cell_57/MatMul/ReadVariableOp?)while/gru_cell_57/MatMul_1/ReadVariableOp? while/gru_cell_57/ReadVariableOp?
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
 while/gru_cell_57/ReadVariableOpReadVariableOp+while_gru_cell_57_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_57/ReadVariableOp?
while/gru_cell_57/unstackUnpack(while/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_57/unstack?
'while/gru_cell_57/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_57_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_57/MatMul/ReadVariableOp?
while/gru_cell_57/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul?
while/gru_cell_57/BiasAddBiasAdd"while/gru_cell_57/MatMul:product:0"while/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd?
!while/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_57/split/split_dim?
while/gru_cell_57/splitSplit*while/gru_cell_57/split/split_dim:output:0"while/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split?
)while/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_57_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_57/MatMul_1/ReadVariableOp?
while/gru_cell_57/MatMul_1MatMulwhile_placeholder_21while/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul_1?
while/gru_cell_57/BiasAdd_1BiasAdd$while/gru_cell_57/MatMul_1:product:0"while/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd_1?
while/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_57/Const?
#while/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_57/split_1/split_dim?
while/gru_cell_57/split_1SplitV$while/gru_cell_57/BiasAdd_1:output:0 while/gru_cell_57/Const:output:0,while/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split_1?
while/gru_cell_57/addAddV2 while/gru_cell_57/split:output:0"while/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add?
while/gru_cell_57/SigmoidSigmoidwhile/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid?
while/gru_cell_57/add_1AddV2 while/gru_cell_57/split:output:1"while/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_1?
while/gru_cell_57/Sigmoid_1Sigmoidwhile/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid_1?
while/gru_cell_57/mulMulwhile/gru_cell_57/Sigmoid_1:y:0"while/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul?
while/gru_cell_57/add_2AddV2 while/gru_cell_57/split:output:2while/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_2?
while/gru_cell_57/ReluReluwhile/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Relu?
while/gru_cell_57/mul_1Mulwhile/gru_cell_57/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_1w
while/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_57/sub/x?
while/gru_cell_57/subSub while/gru_cell_57/sub/x:output:0while/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/sub?
while/gru_cell_57/mul_2Mulwhile/gru_cell_57/sub:z:0$while/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_2?
while/gru_cell_57/add_3AddV2while/gru_cell_57/mul_1:z:0while/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_57/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_57/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_57/MatMul/ReadVariableOp*^while/gru_cell_57/MatMul_1/ReadVariableOp!^while/gru_cell_57/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_57_matmul_1_readvariableop_resource4while_gru_cell_57_matmul_1_readvariableop_resource_0"f
0while_gru_cell_57_matmul_readvariableop_resource2while_gru_cell_57_matmul_readvariableop_resource_0"X
)while_gru_cell_57_readvariableop_resource+while_gru_cell_57_readvariableop_resource_0")
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
'while/gru_cell_57/MatMul/ReadVariableOp'while/gru_cell_57/MatMul/ReadVariableOp2V
)while/gru_cell_57/MatMul_1/ReadVariableOp)while/gru_cell_57/MatMul_1/ReadVariableOp2D
 while/gru_cell_57/ReadVariableOp while/gru_cell_57/ReadVariableOp: 
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
(__inference_gru_57_layer_call_fn_3600605
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
GPU 2J 8? *L
fGRE
C__inference_gru_57_layer_call_and_return_conditional_losses_35976162
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
??
?
#__inference__traced_restore_3601954
file_prefix4
 assignvariableop_dense_63_kernel:
??/
 assignvariableop_1_dense_63_bias:	?6
"assignvariableop_2_dense_64_kernel:
??/
 assignvariableop_3_dense_64_bias:	?5
"assignvariableop_4_dense_65_kernel:	?.
 assignvariableop_5_dense_65_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: @
-assignvariableop_11_gru_56_gru_cell_56_kernel:	?K
7assignvariableop_12_gru_56_gru_cell_56_recurrent_kernel:
??>
+assignvariableop_13_gru_56_gru_cell_56_bias:	?A
-assignvariableop_14_gru_57_gru_cell_57_kernel:
??K
7assignvariableop_15_gru_57_gru_cell_57_recurrent_kernel:
??>
+assignvariableop_16_gru_57_gru_cell_57_bias:	?#
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: >
*assignvariableop_21_adam_dense_63_kernel_m:
??7
(assignvariableop_22_adam_dense_63_bias_m:	?>
*assignvariableop_23_adam_dense_64_kernel_m:
??7
(assignvariableop_24_adam_dense_64_bias_m:	?=
*assignvariableop_25_adam_dense_65_kernel_m:	?6
(assignvariableop_26_adam_dense_65_bias_m:G
4assignvariableop_27_adam_gru_56_gru_cell_56_kernel_m:	?R
>assignvariableop_28_adam_gru_56_gru_cell_56_recurrent_kernel_m:
??E
2assignvariableop_29_adam_gru_56_gru_cell_56_bias_m:	?H
4assignvariableop_30_adam_gru_57_gru_cell_57_kernel_m:
??R
>assignvariableop_31_adam_gru_57_gru_cell_57_recurrent_kernel_m:
??E
2assignvariableop_32_adam_gru_57_gru_cell_57_bias_m:	?>
*assignvariableop_33_adam_dense_63_kernel_v:
??7
(assignvariableop_34_adam_dense_63_bias_v:	?>
*assignvariableop_35_adam_dense_64_kernel_v:
??7
(assignvariableop_36_adam_dense_64_bias_v:	?=
*assignvariableop_37_adam_dense_65_kernel_v:	?6
(assignvariableop_38_adam_dense_65_bias_v:G
4assignvariableop_39_adam_gru_56_gru_cell_56_kernel_v:	?R
>assignvariableop_40_adam_gru_56_gru_cell_56_recurrent_kernel_v:
??E
2assignvariableop_41_adam_gru_56_gru_cell_56_bias_v:	?H
4assignvariableop_42_adam_gru_57_gru_cell_57_kernel_v:
??R
>assignvariableop_43_adam_gru_57_gru_cell_57_recurrent_kernel_v:
??E
2assignvariableop_44_adam_gru_57_gru_cell_57_bias_v:	?
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_63_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_63_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_64_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_64_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_65_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_65_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_gru_56_gru_cell_56_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_gru_56_gru_cell_56_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_gru_56_gru_cell_56_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp-assignvariableop_14_gru_57_gru_cell_57_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp7assignvariableop_15_gru_57_gru_cell_57_recurrent_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_gru_57_gru_cell_57_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_63_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_63_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_64_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_64_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_65_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_65_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_gru_56_gru_cell_56_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_gru_56_gru_cell_56_recurrent_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_gru_56_gru_cell_56_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_gru_57_gru_cell_57_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_gru_57_gru_cell_57_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_gru_57_gru_cell_57_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_63_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_63_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_64_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_64_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_65_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_65_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_gru_56_gru_cell_56_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp>assignvariableop_40_adam_gru_56_gru_cell_56_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_gru_56_gru_cell_56_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adam_gru_57_gru_cell_57_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp>assignvariableop_43_adam_gru_57_gru_cell_57_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp2assignvariableop_44_adam_gru_57_gru_cell_57_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45f
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_46?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
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
?
H
,__inference_dropout_94_layer_call_fn_3601378

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
GPU 2J 8? *P
fKRI
G__inference_dropout_94_layer_call_and_return_conditional_losses_35982692
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
?X
?
C__inference_gru_56_layer_call_and_return_conditional_losses_3598001

inputs6
#gru_cell_56_readvariableop_resource:	?=
*gru_cell_56_matmul_readvariableop_resource:	?@
,gru_cell_56_matmul_1_readvariableop_resource:
??
identity??!gru_cell_56/MatMul/ReadVariableOp?#gru_cell_56/MatMul_1/ReadVariableOp?gru_cell_56/ReadVariableOp?whileD
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
gru_cell_56/ReadVariableOpReadVariableOp#gru_cell_56_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_56/ReadVariableOp?
gru_cell_56/unstackUnpack"gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_56/unstack?
!gru_cell_56/MatMul/ReadVariableOpReadVariableOp*gru_cell_56_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_56/MatMul/ReadVariableOp?
gru_cell_56/MatMulMatMulstrided_slice_2:output:0)gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul?
gru_cell_56/BiasAddBiasAddgru_cell_56/MatMul:product:0gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd?
gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split/split_dim?
gru_cell_56/splitSplit$gru_cell_56/split/split_dim:output:0gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split?
#gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_56_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_56/MatMul_1/ReadVariableOp?
gru_cell_56/MatMul_1MatMulzeros:output:0+gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul_1?
gru_cell_56/BiasAdd_1BiasAddgru_cell_56/MatMul_1:product:0gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd_1{
gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_56/Const?
gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split_1/split_dim?
gru_cell_56/split_1SplitVgru_cell_56/BiasAdd_1:output:0gru_cell_56/Const:output:0&gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split_1?
gru_cell_56/addAddV2gru_cell_56/split:output:0gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add}
gru_cell_56/SigmoidSigmoidgru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid?
gru_cell_56/add_1AddV2gru_cell_56/split:output:1gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_1?
gru_cell_56/Sigmoid_1Sigmoidgru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid_1?
gru_cell_56/mulMulgru_cell_56/Sigmoid_1:y:0gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul?
gru_cell_56/add_2AddV2gru_cell_56/split:output:2gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_2v
gru_cell_56/ReluRelugru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Relu?
gru_cell_56/mul_1Mulgru_cell_56/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_1k
gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_56/sub/x?
gru_cell_56/subSubgru_cell_56/sub/x:output:0gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/sub?
gru_cell_56/mul_2Mulgru_cell_56/sub:z:0gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_2?
gru_cell_56/add_3AddV2gru_cell_56/mul_1:z:0gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_56_readvariableop_resource*gru_cell_56_matmul_readvariableop_resource,gru_cell_56_matmul_1_readvariableop_resource*
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
bodyR
while_body_3597912*
condR
while_cond_3597911*9
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
NoOpNoOp"^gru_cell_56/MatMul/ReadVariableOp$^gru_cell_56/MatMul_1/ReadVariableOp^gru_cell_56/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_56/MatMul/ReadVariableOp!gru_cell_56/MatMul/ReadVariableOp2J
#gru_cell_56/MatMul_1/ReadVariableOp#gru_cell_56/MatMul_1/ReadVariableOp28
gru_cell_56/ReadVariableOpgru_cell_56/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?X
?
C__inference_gru_57_layer_call_and_return_conditional_losses_3601086

inputs6
#gru_cell_57_readvariableop_resource:	?>
*gru_cell_57_matmul_readvariableop_resource:
??@
,gru_cell_57_matmul_1_readvariableop_resource:
??
identity??!gru_cell_57/MatMul/ReadVariableOp?#gru_cell_57/MatMul_1/ReadVariableOp?gru_cell_57/ReadVariableOp?whileD
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
gru_cell_57/ReadVariableOpReadVariableOp#gru_cell_57_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_57/ReadVariableOp?
gru_cell_57/unstackUnpack"gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_57/unstack?
!gru_cell_57/MatMul/ReadVariableOpReadVariableOp*gru_cell_57_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_57/MatMul/ReadVariableOp?
gru_cell_57/MatMulMatMulstrided_slice_2:output:0)gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul?
gru_cell_57/BiasAddBiasAddgru_cell_57/MatMul:product:0gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd?
gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split/split_dim?
gru_cell_57/splitSplit$gru_cell_57/split/split_dim:output:0gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split?
#gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_57_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_57/MatMul_1/ReadVariableOp?
gru_cell_57/MatMul_1MatMulzeros:output:0+gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul_1?
gru_cell_57/BiasAdd_1BiasAddgru_cell_57/MatMul_1:product:0gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd_1{
gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_57/Const?
gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split_1/split_dim?
gru_cell_57/split_1SplitVgru_cell_57/BiasAdd_1:output:0gru_cell_57/Const:output:0&gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split_1?
gru_cell_57/addAddV2gru_cell_57/split:output:0gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add}
gru_cell_57/SigmoidSigmoidgru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid?
gru_cell_57/add_1AddV2gru_cell_57/split:output:1gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_1?
gru_cell_57/Sigmoid_1Sigmoidgru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid_1?
gru_cell_57/mulMulgru_cell_57/Sigmoid_1:y:0gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul?
gru_cell_57/add_2AddV2gru_cell_57/split:output:2gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_2v
gru_cell_57/ReluRelugru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Relu?
gru_cell_57/mul_1Mulgru_cell_57/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_1k
gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_57/sub/x?
gru_cell_57/subSubgru_cell_57/sub/x:output:0gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/sub?
gru_cell_57/mul_2Mulgru_cell_57/sub:z:0gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_2?
gru_cell_57/add_3AddV2gru_cell_57/mul_1:z:0gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_57_readvariableop_resource*gru_cell_57_matmul_readvariableop_resource,gru_cell_57_matmul_1_readvariableop_resource*
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
bodyR
while_body_3600997*
condR
while_cond_3600996*9
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
NoOpNoOp"^gru_cell_57/MatMul/ReadVariableOp$^gru_cell_57/MatMul_1/ReadVariableOp^gru_cell_57/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_57/MatMul/ReadVariableOp!gru_cell_57/MatMul/ReadVariableOp2J
#gru_cell_57/MatMul_1/ReadVariableOp#gru_cell_57/MatMul_1/ReadVariableOp28
gru_cell_57/ReadVariableOpgru_cell_57/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
E__inference_dense_63_layer_call_and_return_conditional_losses_3601306

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
?
e
G__inference_dropout_94_layer_call_and_return_conditional_losses_3601388

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
e
,__inference_dropout_91_layer_call_fn_3600566

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
GPU 2J 8? *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_35986292
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
(__inference_gru_57_layer_call_fn_3600616

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
GPU 2J 8? *L
fGRE
C__inference_gru_57_layer_call_and_return_conditional_losses_35981682
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
?
H
,__inference_dropout_92_layer_call_fn_3601244

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
GPU 2J 8? *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_35981812
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
e
G__inference_dropout_93_layer_call_and_return_conditional_losses_3601321

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
(__inference_gru_56_layer_call_fn_3599933

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
GPU 2J 8? *L
fGRE
C__inference_gru_56_layer_call_and_return_conditional_losses_35980012
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
?	
?
gru_57_while_cond_3599313*
&gru_57_while_gru_57_while_loop_counter0
,gru_57_while_gru_57_while_maximum_iterations
gru_57_while_placeholder
gru_57_while_placeholder_1
gru_57_while_placeholder_2,
(gru_57_while_less_gru_57_strided_slice_1C
?gru_57_while_gru_57_while_cond_3599313___redundant_placeholder0C
?gru_57_while_gru_57_while_cond_3599313___redundant_placeholder1C
?gru_57_while_gru_57_while_cond_3599313___redundant_placeholder2C
?gru_57_while_gru_57_while_cond_3599313___redundant_placeholder3
gru_57_while_identity
?
gru_57/while/LessLessgru_57_while_placeholder(gru_57_while_less_gru_57_strided_slice_1*
T0*
_output_shapes
: 2
gru_57/while/Lessr
gru_57/while/IdentityIdentitygru_57/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_57/while/Identity"7
gru_57_while_identitygru_57/while/Identity:output:0*(
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
e
G__inference_dropout_91_layer_call_and_return_conditional_losses_3598014

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
e
G__inference_dropout_93_layer_call_and_return_conditional_losses_3598225

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
?
while_cond_3596985
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3596985___redundant_placeholder05
1while_while_cond_3596985___redundant_placeholder15
1while_while_cond_3596985___redundant_placeholder25
1while_while_cond_3596985___redundant_placeholder3
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
C__inference_gru_56_layer_call_and_return_conditional_losses_3600097
inputs_06
#gru_cell_56_readvariableop_resource:	?=
*gru_cell_56_matmul_readvariableop_resource:	?@
,gru_cell_56_matmul_1_readvariableop_resource:
??
identity??!gru_cell_56/MatMul/ReadVariableOp?#gru_cell_56/MatMul_1/ReadVariableOp?gru_cell_56/ReadVariableOp?whileF
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
gru_cell_56/ReadVariableOpReadVariableOp#gru_cell_56_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_56/ReadVariableOp?
gru_cell_56/unstackUnpack"gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_56/unstack?
!gru_cell_56/MatMul/ReadVariableOpReadVariableOp*gru_cell_56_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_56/MatMul/ReadVariableOp?
gru_cell_56/MatMulMatMulstrided_slice_2:output:0)gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul?
gru_cell_56/BiasAddBiasAddgru_cell_56/MatMul:product:0gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd?
gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split/split_dim?
gru_cell_56/splitSplit$gru_cell_56/split/split_dim:output:0gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split?
#gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_56_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_56/MatMul_1/ReadVariableOp?
gru_cell_56/MatMul_1MatMulzeros:output:0+gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul_1?
gru_cell_56/BiasAdd_1BiasAddgru_cell_56/MatMul_1:product:0gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd_1{
gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_56/Const?
gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split_1/split_dim?
gru_cell_56/split_1SplitVgru_cell_56/BiasAdd_1:output:0gru_cell_56/Const:output:0&gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split_1?
gru_cell_56/addAddV2gru_cell_56/split:output:0gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add}
gru_cell_56/SigmoidSigmoidgru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid?
gru_cell_56/add_1AddV2gru_cell_56/split:output:1gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_1?
gru_cell_56/Sigmoid_1Sigmoidgru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid_1?
gru_cell_56/mulMulgru_cell_56/Sigmoid_1:y:0gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul?
gru_cell_56/add_2AddV2gru_cell_56/split:output:2gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_2v
gru_cell_56/ReluRelugru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Relu?
gru_cell_56/mul_1Mulgru_cell_56/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_1k
gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_56/sub/x?
gru_cell_56/subSubgru_cell_56/sub/x:output:0gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/sub?
gru_cell_56/mul_2Mulgru_cell_56/sub:z:0gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_2?
gru_cell_56/add_3AddV2gru_cell_56/mul_1:z:0gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_56_readvariableop_resource*gru_cell_56_matmul_readvariableop_resource,gru_cell_56_matmul_1_readvariableop_resource*
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
bodyR
while_body_3600008*
condR
while_cond_3600007*9
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
NoOpNoOp"^gru_cell_56/MatMul/ReadVariableOp$^gru_cell_56/MatMul_1/ReadVariableOp^gru_cell_56/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2F
!gru_cell_56/MatMul/ReadVariableOp!gru_cell_56/MatMul/ReadVariableOp2J
#gru_cell_56/MatMul_1/ReadVariableOp#gru_cell_56/MatMul_1/ReadVariableOp28
gru_cell_56/ReadVariableOpgru_cell_56/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
'sequential_28_gru_57_while_cond_3596537F
Bsequential_28_gru_57_while_sequential_28_gru_57_while_loop_counterL
Hsequential_28_gru_57_while_sequential_28_gru_57_while_maximum_iterations*
&sequential_28_gru_57_while_placeholder,
(sequential_28_gru_57_while_placeholder_1,
(sequential_28_gru_57_while_placeholder_2H
Dsequential_28_gru_57_while_less_sequential_28_gru_57_strided_slice_1_
[sequential_28_gru_57_while_sequential_28_gru_57_while_cond_3596537___redundant_placeholder0_
[sequential_28_gru_57_while_sequential_28_gru_57_while_cond_3596537___redundant_placeholder1_
[sequential_28_gru_57_while_sequential_28_gru_57_while_cond_3596537___redundant_placeholder2_
[sequential_28_gru_57_while_sequential_28_gru_57_while_cond_3596537___redundant_placeholder3'
#sequential_28_gru_57_while_identity
?
sequential_28/gru_57/while/LessLess&sequential_28_gru_57_while_placeholderDsequential_28_gru_57_while_less_sequential_28_gru_57_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_28/gru_57/while/Less?
#sequential_28/gru_57/while/IdentityIdentity#sequential_28/gru_57/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_28/gru_57/while/Identity"S
#sequential_28_gru_57_while_identity,sequential_28/gru_57/while/Identity:output:0*(
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
?P
?	
gru_57_while_body_3599314*
&gru_57_while_gru_57_while_loop_counter0
,gru_57_while_gru_57_while_maximum_iterations
gru_57_while_placeholder
gru_57_while_placeholder_1
gru_57_while_placeholder_2)
%gru_57_while_gru_57_strided_slice_1_0e
agru_57_while_tensorarrayv2read_tensorlistgetitem_gru_57_tensorarrayunstack_tensorlistfromtensor_0E
2gru_57_while_gru_cell_57_readvariableop_resource_0:	?M
9gru_57_while_gru_cell_57_matmul_readvariableop_resource_0:
??O
;gru_57_while_gru_cell_57_matmul_1_readvariableop_resource_0:
??
gru_57_while_identity
gru_57_while_identity_1
gru_57_while_identity_2
gru_57_while_identity_3
gru_57_while_identity_4'
#gru_57_while_gru_57_strided_slice_1c
_gru_57_while_tensorarrayv2read_tensorlistgetitem_gru_57_tensorarrayunstack_tensorlistfromtensorC
0gru_57_while_gru_cell_57_readvariableop_resource:	?K
7gru_57_while_gru_cell_57_matmul_readvariableop_resource:
??M
9gru_57_while_gru_cell_57_matmul_1_readvariableop_resource:
????.gru_57/while/gru_cell_57/MatMul/ReadVariableOp?0gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp?'gru_57/while/gru_cell_57/ReadVariableOp?
>gru_57/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_57/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_57/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_57_while_tensorarrayv2read_tensorlistgetitem_gru_57_tensorarrayunstack_tensorlistfromtensor_0gru_57_while_placeholderGgru_57/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0gru_57/while/TensorArrayV2Read/TensorListGetItem?
'gru_57/while/gru_cell_57/ReadVariableOpReadVariableOp2gru_57_while_gru_cell_57_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_57/while/gru_cell_57/ReadVariableOp?
 gru_57/while/gru_cell_57/unstackUnpack/gru_57/while/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_57/while/gru_cell_57/unstack?
.gru_57/while/gru_cell_57/MatMul/ReadVariableOpReadVariableOp9gru_57_while_gru_cell_57_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype020
.gru_57/while/gru_cell_57/MatMul/ReadVariableOp?
gru_57/while/gru_cell_57/MatMulMatMul7gru_57/while/TensorArrayV2Read/TensorListGetItem:item:06gru_57/while/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_57/while/gru_cell_57/MatMul?
 gru_57/while/gru_cell_57/BiasAddBiasAdd)gru_57/while/gru_cell_57/MatMul:product:0)gru_57/while/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_57/while/gru_cell_57/BiasAdd?
(gru_57/while/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_57/while/gru_cell_57/split/split_dim?
gru_57/while/gru_cell_57/splitSplit1gru_57/while/gru_cell_57/split/split_dim:output:0)gru_57/while/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_57/while/gru_cell_57/split?
0gru_57/while/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp;gru_57_while_gru_cell_57_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp?
!gru_57/while/gru_cell_57/MatMul_1MatMulgru_57_while_placeholder_28gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_57/while/gru_cell_57/MatMul_1?
"gru_57/while/gru_cell_57/BiasAdd_1BiasAdd+gru_57/while/gru_cell_57/MatMul_1:product:0)gru_57/while/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_57/while/gru_cell_57/BiasAdd_1?
gru_57/while/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2 
gru_57/while/gru_cell_57/Const?
*gru_57/while/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_57/while/gru_cell_57/split_1/split_dim?
 gru_57/while/gru_cell_57/split_1SplitV+gru_57/while/gru_cell_57/BiasAdd_1:output:0'gru_57/while/gru_cell_57/Const:output:03gru_57/while/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_57/while/gru_cell_57/split_1?
gru_57/while/gru_cell_57/addAddV2'gru_57/while/gru_cell_57/split:output:0)gru_57/while/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_57/while/gru_cell_57/add?
 gru_57/while/gru_cell_57/SigmoidSigmoid gru_57/while/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_57/while/gru_cell_57/Sigmoid?
gru_57/while/gru_cell_57/add_1AddV2'gru_57/while/gru_cell_57/split:output:1)gru_57/while/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_57/while/gru_cell_57/add_1?
"gru_57/while/gru_cell_57/Sigmoid_1Sigmoid"gru_57/while/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_57/while/gru_cell_57/Sigmoid_1?
gru_57/while/gru_cell_57/mulMul&gru_57/while/gru_cell_57/Sigmoid_1:y:0)gru_57/while/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_57/while/gru_cell_57/mul?
gru_57/while/gru_cell_57/add_2AddV2'gru_57/while/gru_cell_57/split:output:2 gru_57/while/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_57/while/gru_cell_57/add_2?
gru_57/while/gru_cell_57/ReluRelu"gru_57/while/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_57/while/gru_cell_57/Relu?
gru_57/while/gru_cell_57/mul_1Mul$gru_57/while/gru_cell_57/Sigmoid:y:0gru_57_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_57/while/gru_cell_57/mul_1?
gru_57/while/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_57/while/gru_cell_57/sub/x?
gru_57/while/gru_cell_57/subSub'gru_57/while/gru_cell_57/sub/x:output:0$gru_57/while/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_57/while/gru_cell_57/sub?
gru_57/while/gru_cell_57/mul_2Mul gru_57/while/gru_cell_57/sub:z:0+gru_57/while/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_57/while/gru_cell_57/mul_2?
gru_57/while/gru_cell_57/add_3AddV2"gru_57/while/gru_cell_57/mul_1:z:0"gru_57/while/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_57/while/gru_cell_57/add_3?
1gru_57/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_57_while_placeholder_1gru_57_while_placeholder"gru_57/while/gru_cell_57/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_57/while/TensorArrayV2Write/TensorListSetItemj
gru_57/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_57/while/add/y?
gru_57/while/addAddV2gru_57_while_placeholdergru_57/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_57/while/addn
gru_57/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_57/while/add_1/y?
gru_57/while/add_1AddV2&gru_57_while_gru_57_while_loop_countergru_57/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_57/while/add_1?
gru_57/while/IdentityIdentitygru_57/while/add_1:z:0^gru_57/while/NoOp*
T0*
_output_shapes
: 2
gru_57/while/Identity?
gru_57/while/Identity_1Identity,gru_57_while_gru_57_while_maximum_iterations^gru_57/while/NoOp*
T0*
_output_shapes
: 2
gru_57/while/Identity_1?
gru_57/while/Identity_2Identitygru_57/while/add:z:0^gru_57/while/NoOp*
T0*
_output_shapes
: 2
gru_57/while/Identity_2?
gru_57/while/Identity_3IdentityAgru_57/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_57/while/NoOp*
T0*
_output_shapes
: 2
gru_57/while/Identity_3?
gru_57/while/Identity_4Identity"gru_57/while/gru_cell_57/add_3:z:0^gru_57/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_57/while/Identity_4?
gru_57/while/NoOpNoOp/^gru_57/while/gru_cell_57/MatMul/ReadVariableOp1^gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp(^gru_57/while/gru_cell_57/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_57/while/NoOp"L
#gru_57_while_gru_57_strided_slice_1%gru_57_while_gru_57_strided_slice_1_0"x
9gru_57_while_gru_cell_57_matmul_1_readvariableop_resource;gru_57_while_gru_cell_57_matmul_1_readvariableop_resource_0"t
7gru_57_while_gru_cell_57_matmul_readvariableop_resource9gru_57_while_gru_cell_57_matmul_readvariableop_resource_0"f
0gru_57_while_gru_cell_57_readvariableop_resource2gru_57_while_gru_cell_57_readvariableop_resource_0"7
gru_57_while_identitygru_57/while/Identity:output:0";
gru_57_while_identity_1 gru_57/while/Identity_1:output:0";
gru_57_while_identity_2 gru_57/while/Identity_2:output:0";
gru_57_while_identity_3 gru_57/while/Identity_3:output:0";
gru_57_while_identity_4 gru_57/while/Identity_4:output:0"?
_gru_57_while_tensorarrayv2read_tensorlistgetitem_gru_57_tensorarrayunstack_tensorlistfromtensoragru_57_while_tensorarrayv2read_tensorlistgetitem_gru_57_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_57/while/gru_cell_57/MatMul/ReadVariableOp.gru_57/while/gru_cell_57/MatMul/ReadVariableOp2d
0gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp0gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp2R
'gru_57/while/gru_cell_57/ReadVariableOp'gru_57/while/gru_cell_57/ReadVariableOp: 
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
while_body_3600161
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_56_readvariableop_resource_0:	?E
2while_gru_cell_56_matmul_readvariableop_resource_0:	?H
4while_gru_cell_56_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_56_readvariableop_resource:	?C
0while_gru_cell_56_matmul_readvariableop_resource:	?F
2while_gru_cell_56_matmul_1_readvariableop_resource:
????'while/gru_cell_56/MatMul/ReadVariableOp?)while/gru_cell_56/MatMul_1/ReadVariableOp? while/gru_cell_56/ReadVariableOp?
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
 while/gru_cell_56/ReadVariableOpReadVariableOp+while_gru_cell_56_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_56/ReadVariableOp?
while/gru_cell_56/unstackUnpack(while/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_56/unstack?
'while/gru_cell_56/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_56_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_56/MatMul/ReadVariableOp?
while/gru_cell_56/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul?
while/gru_cell_56/BiasAddBiasAdd"while/gru_cell_56/MatMul:product:0"while/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd?
!while/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_56/split/split_dim?
while/gru_cell_56/splitSplit*while/gru_cell_56/split/split_dim:output:0"while/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split?
)while/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_56_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_56/MatMul_1/ReadVariableOp?
while/gru_cell_56/MatMul_1MatMulwhile_placeholder_21while/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul_1?
while/gru_cell_56/BiasAdd_1BiasAdd$while/gru_cell_56/MatMul_1:product:0"while/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd_1?
while/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_56/Const?
#while/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_56/split_1/split_dim?
while/gru_cell_56/split_1SplitV$while/gru_cell_56/BiasAdd_1:output:0 while/gru_cell_56/Const:output:0,while/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split_1?
while/gru_cell_56/addAddV2 while/gru_cell_56/split:output:0"while/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add?
while/gru_cell_56/SigmoidSigmoidwhile/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid?
while/gru_cell_56/add_1AddV2 while/gru_cell_56/split:output:1"while/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_1?
while/gru_cell_56/Sigmoid_1Sigmoidwhile/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid_1?
while/gru_cell_56/mulMulwhile/gru_cell_56/Sigmoid_1:y:0"while/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul?
while/gru_cell_56/add_2AddV2 while/gru_cell_56/split:output:2while/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_2?
while/gru_cell_56/ReluReluwhile/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Relu?
while/gru_cell_56/mul_1Mulwhile/gru_cell_56/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_1w
while/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_56/sub/x?
while/gru_cell_56/subSub while/gru_cell_56/sub/x:output:0while/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/sub?
while/gru_cell_56/mul_2Mulwhile/gru_cell_56/sub:z:0$while/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_2?
while/gru_cell_56/add_3AddV2while/gru_cell_56/mul_1:z:0while/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_56/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_56/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_56/MatMul/ReadVariableOp*^while/gru_cell_56/MatMul_1/ReadVariableOp!^while/gru_cell_56/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_56_matmul_1_readvariableop_resource4while_gru_cell_56_matmul_1_readvariableop_resource_0"f
0while_gru_cell_56_matmul_readvariableop_resource2while_gru_cell_56_matmul_readvariableop_resource_0"X
)while_gru_cell_56_readvariableop_resource+while_gru_cell_56_readvariableop_resource_0")
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
'while/gru_cell_56/MatMul/ReadVariableOp'while/gru_cell_56/MatMul/ReadVariableOp2V
)while/gru_cell_56/MatMul_1/ReadVariableOp)while/gru_cell_56/MatMul_1/ReadVariableOp2D
 while/gru_cell_56/ReadVariableOp while/gru_cell_56/ReadVariableOp: 
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
?_
?
 __inference__traced_save_3601809
file_prefix.
*savev2_dense_63_kernel_read_readvariableop,
(savev2_dense_63_bias_read_readvariableop.
*savev2_dense_64_kernel_read_readvariableop,
(savev2_dense_64_bias_read_readvariableop.
*savev2_dense_65_kernel_read_readvariableop,
(savev2_dense_65_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_gru_56_gru_cell_56_kernel_read_readvariableopB
>savev2_gru_56_gru_cell_56_recurrent_kernel_read_readvariableop6
2savev2_gru_56_gru_cell_56_bias_read_readvariableop8
4savev2_gru_57_gru_cell_57_kernel_read_readvariableopB
>savev2_gru_57_gru_cell_57_recurrent_kernel_read_readvariableop6
2savev2_gru_57_gru_cell_57_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_63_kernel_m_read_readvariableop3
/savev2_adam_dense_63_bias_m_read_readvariableop5
1savev2_adam_dense_64_kernel_m_read_readvariableop3
/savev2_adam_dense_64_bias_m_read_readvariableop5
1savev2_adam_dense_65_kernel_m_read_readvariableop3
/savev2_adam_dense_65_bias_m_read_readvariableop?
;savev2_adam_gru_56_gru_cell_56_kernel_m_read_readvariableopI
Esavev2_adam_gru_56_gru_cell_56_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_56_gru_cell_56_bias_m_read_readvariableop?
;savev2_adam_gru_57_gru_cell_57_kernel_m_read_readvariableopI
Esavev2_adam_gru_57_gru_cell_57_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_57_gru_cell_57_bias_m_read_readvariableop5
1savev2_adam_dense_63_kernel_v_read_readvariableop3
/savev2_adam_dense_63_bias_v_read_readvariableop5
1savev2_adam_dense_64_kernel_v_read_readvariableop3
/savev2_adam_dense_64_bias_v_read_readvariableop5
1savev2_adam_dense_65_kernel_v_read_readvariableop3
/savev2_adam_dense_65_bias_v_read_readvariableop?
;savev2_adam_gru_56_gru_cell_56_kernel_v_read_readvariableopI
Esavev2_adam_gru_56_gru_cell_56_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_56_gru_cell_56_bias_v_read_readvariableop?
;savev2_adam_gru_57_gru_cell_57_kernel_v_read_readvariableopI
Esavev2_adam_gru_57_gru_cell_57_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_57_gru_cell_57_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_63_kernel_read_readvariableop(savev2_dense_63_bias_read_readvariableop*savev2_dense_64_kernel_read_readvariableop(savev2_dense_64_bias_read_readvariableop*savev2_dense_65_kernel_read_readvariableop(savev2_dense_65_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_gru_56_gru_cell_56_kernel_read_readvariableop>savev2_gru_56_gru_cell_56_recurrent_kernel_read_readvariableop2savev2_gru_56_gru_cell_56_bias_read_readvariableop4savev2_gru_57_gru_cell_57_kernel_read_readvariableop>savev2_gru_57_gru_cell_57_recurrent_kernel_read_readvariableop2savev2_gru_57_gru_cell_57_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_63_kernel_m_read_readvariableop/savev2_adam_dense_63_bias_m_read_readvariableop1savev2_adam_dense_64_kernel_m_read_readvariableop/savev2_adam_dense_64_bias_m_read_readvariableop1savev2_adam_dense_65_kernel_m_read_readvariableop/savev2_adam_dense_65_bias_m_read_readvariableop;savev2_adam_gru_56_gru_cell_56_kernel_m_read_readvariableopEsavev2_adam_gru_56_gru_cell_56_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_56_gru_cell_56_bias_m_read_readvariableop;savev2_adam_gru_57_gru_cell_57_kernel_m_read_readvariableopEsavev2_adam_gru_57_gru_cell_57_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_57_gru_cell_57_bias_m_read_readvariableop1savev2_adam_dense_63_kernel_v_read_readvariableop/savev2_adam_dense_63_bias_v_read_readvariableop1savev2_adam_dense_64_kernel_v_read_readvariableop/savev2_adam_dense_64_bias_v_read_readvariableop1savev2_adam_dense_65_kernel_v_read_readvariableop/savev2_adam_dense_65_bias_v_read_readvariableop;savev2_adam_gru_56_gru_cell_56_kernel_v_read_readvariableopEsavev2_adam_gru_56_gru_cell_56_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_56_gru_cell_56_bias_v_read_readvariableop;savev2_adam_gru_57_gru_cell_57_kernel_v_read_readvariableopEsavev2_adam_gru_57_gru_cell_57_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_57_gru_cell_57_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:	?:: : : : : :	?:
??:	?:
??:
??:	?: : : : :
??:?:
??:?:	?::	?:
??:	?:
??:
??:	?:
??:?:
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
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:& "
 
_output_shapes
:
??:%!!

_output_shapes
:	?:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?: '

_output_shapes
::%(!

_output_shapes
:	?:&)"
 
_output_shapes
:
??:%*!

_output_shapes
:	?:&+"
 
_output_shapes
:
??:&,"
 
_output_shapes
:
??:%-!

_output_shapes
:	?:.

_output_shapes
: 
?Y
?
C__inference_gru_57_layer_call_and_return_conditional_losses_3600780
inputs_06
#gru_cell_57_readvariableop_resource:	?>
*gru_cell_57_matmul_readvariableop_resource:
??@
,gru_cell_57_matmul_1_readvariableop_resource:
??
identity??!gru_cell_57/MatMul/ReadVariableOp?#gru_cell_57/MatMul_1/ReadVariableOp?gru_cell_57/ReadVariableOp?whileF
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
gru_cell_57/ReadVariableOpReadVariableOp#gru_cell_57_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_57/ReadVariableOp?
gru_cell_57/unstackUnpack"gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_57/unstack?
!gru_cell_57/MatMul/ReadVariableOpReadVariableOp*gru_cell_57_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_57/MatMul/ReadVariableOp?
gru_cell_57/MatMulMatMulstrided_slice_2:output:0)gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul?
gru_cell_57/BiasAddBiasAddgru_cell_57/MatMul:product:0gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd?
gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split/split_dim?
gru_cell_57/splitSplit$gru_cell_57/split/split_dim:output:0gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split?
#gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_57_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_57/MatMul_1/ReadVariableOp?
gru_cell_57/MatMul_1MatMulzeros:output:0+gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul_1?
gru_cell_57/BiasAdd_1BiasAddgru_cell_57/MatMul_1:product:0gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd_1{
gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_57/Const?
gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split_1/split_dim?
gru_cell_57/split_1SplitVgru_cell_57/BiasAdd_1:output:0gru_cell_57/Const:output:0&gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split_1?
gru_cell_57/addAddV2gru_cell_57/split:output:0gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add}
gru_cell_57/SigmoidSigmoidgru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid?
gru_cell_57/add_1AddV2gru_cell_57/split:output:1gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_1?
gru_cell_57/Sigmoid_1Sigmoidgru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid_1?
gru_cell_57/mulMulgru_cell_57/Sigmoid_1:y:0gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul?
gru_cell_57/add_2AddV2gru_cell_57/split:output:2gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_2v
gru_cell_57/ReluRelugru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Relu?
gru_cell_57/mul_1Mulgru_cell_57/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_1k
gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_57/sub/x?
gru_cell_57/subSubgru_cell_57/sub/x:output:0gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/sub?
gru_cell_57/mul_2Mulgru_cell_57/sub:z:0gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_2?
gru_cell_57/add_3AddV2gru_cell_57/mul_1:z:0gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_57_readvariableop_resource*gru_cell_57_matmul_readvariableop_resource,gru_cell_57_matmul_1_readvariableop_resource*
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
bodyR
while_body_3600691*
condR
while_cond_3600690*9
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
NoOpNoOp"^gru_cell_57/MatMul/ReadVariableOp$^gru_cell_57/MatMul_1/ReadVariableOp^gru_cell_57/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2F
!gru_cell_57/MatMul/ReadVariableOp!gru_cell_57/MatMul/ReadVariableOp2J
#gru_cell_57/MatMul_1/ReadVariableOp#gru_cell_57/MatMul_1/ReadVariableOp28
gru_cell_57/ReadVariableOpgru_cell_57/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?E
?
while_body_3600997
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_57_readvariableop_resource_0:	?F
2while_gru_cell_57_matmul_readvariableop_resource_0:
??H
4while_gru_cell_57_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_57_readvariableop_resource:	?D
0while_gru_cell_57_matmul_readvariableop_resource:
??F
2while_gru_cell_57_matmul_1_readvariableop_resource:
????'while/gru_cell_57/MatMul/ReadVariableOp?)while/gru_cell_57/MatMul_1/ReadVariableOp? while/gru_cell_57/ReadVariableOp?
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
 while/gru_cell_57/ReadVariableOpReadVariableOp+while_gru_cell_57_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_57/ReadVariableOp?
while/gru_cell_57/unstackUnpack(while/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_57/unstack?
'while/gru_cell_57/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_57_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_57/MatMul/ReadVariableOp?
while/gru_cell_57/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul?
while/gru_cell_57/BiasAddBiasAdd"while/gru_cell_57/MatMul:product:0"while/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd?
!while/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_57/split/split_dim?
while/gru_cell_57/splitSplit*while/gru_cell_57/split/split_dim:output:0"while/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split?
)while/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_57_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_57/MatMul_1/ReadVariableOp?
while/gru_cell_57/MatMul_1MatMulwhile_placeholder_21while/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul_1?
while/gru_cell_57/BiasAdd_1BiasAdd$while/gru_cell_57/MatMul_1:product:0"while/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd_1?
while/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_57/Const?
#while/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_57/split_1/split_dim?
while/gru_cell_57/split_1SplitV$while/gru_cell_57/BiasAdd_1:output:0 while/gru_cell_57/Const:output:0,while/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split_1?
while/gru_cell_57/addAddV2 while/gru_cell_57/split:output:0"while/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add?
while/gru_cell_57/SigmoidSigmoidwhile/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid?
while/gru_cell_57/add_1AddV2 while/gru_cell_57/split:output:1"while/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_1?
while/gru_cell_57/Sigmoid_1Sigmoidwhile/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid_1?
while/gru_cell_57/mulMulwhile/gru_cell_57/Sigmoid_1:y:0"while/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul?
while/gru_cell_57/add_2AddV2 while/gru_cell_57/split:output:2while/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_2?
while/gru_cell_57/ReluReluwhile/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Relu?
while/gru_cell_57/mul_1Mulwhile/gru_cell_57/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_1w
while/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_57/sub/x?
while/gru_cell_57/subSub while/gru_cell_57/sub/x:output:0while/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/sub?
while/gru_cell_57/mul_2Mulwhile/gru_cell_57/sub:z:0$while/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_2?
while/gru_cell_57/add_3AddV2while/gru_cell_57/mul_1:z:0while/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_57/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_57/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_57/MatMul/ReadVariableOp*^while/gru_cell_57/MatMul_1/ReadVariableOp!^while/gru_cell_57/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_57_matmul_1_readvariableop_resource4while_gru_cell_57_matmul_1_readvariableop_resource_0"f
0while_gru_cell_57_matmul_readvariableop_resource2while_gru_cell_57_matmul_readvariableop_resource_0"X
)while_gru_cell_57_readvariableop_resource+while_gru_cell_57_readvariableop_resource_0")
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
'while/gru_cell_57/MatMul/ReadVariableOp'while/gru_cell_57/MatMul/ReadVariableOp2V
)while/gru_cell_57/MatMul_1/ReadVariableOp)while/gru_cell_57/MatMul_1/ReadVariableOp2D
 while/gru_cell_57/ReadVariableOp while/gru_cell_57/ReadVariableOp: 
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
?
%__inference_signature_wrapper_3599042
gru_56_input
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
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_56_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_35967102
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
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_56_input
?!
?
E__inference_dense_64_layer_call_and_return_conditional_losses_3601373

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
?X
?
C__inference_gru_57_layer_call_and_return_conditional_losses_3598168

inputs6
#gru_cell_57_readvariableop_resource:	?>
*gru_cell_57_matmul_readvariableop_resource:
??@
,gru_cell_57_matmul_1_readvariableop_resource:
??
identity??!gru_cell_57/MatMul/ReadVariableOp?#gru_cell_57/MatMul_1/ReadVariableOp?gru_cell_57/ReadVariableOp?whileD
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
gru_cell_57/ReadVariableOpReadVariableOp#gru_cell_57_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_57/ReadVariableOp?
gru_cell_57/unstackUnpack"gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_57/unstack?
!gru_cell_57/MatMul/ReadVariableOpReadVariableOp*gru_cell_57_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_57/MatMul/ReadVariableOp?
gru_cell_57/MatMulMatMulstrided_slice_2:output:0)gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul?
gru_cell_57/BiasAddBiasAddgru_cell_57/MatMul:product:0gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd?
gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split/split_dim?
gru_cell_57/splitSplit$gru_cell_57/split/split_dim:output:0gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split?
#gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_57_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_57/MatMul_1/ReadVariableOp?
gru_cell_57/MatMul_1MatMulzeros:output:0+gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul_1?
gru_cell_57/BiasAdd_1BiasAddgru_cell_57/MatMul_1:product:0gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd_1{
gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_57/Const?
gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split_1/split_dim?
gru_cell_57/split_1SplitVgru_cell_57/BiasAdd_1:output:0gru_cell_57/Const:output:0&gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split_1?
gru_cell_57/addAddV2gru_cell_57/split:output:0gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add}
gru_cell_57/SigmoidSigmoidgru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid?
gru_cell_57/add_1AddV2gru_cell_57/split:output:1gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_1?
gru_cell_57/Sigmoid_1Sigmoidgru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid_1?
gru_cell_57/mulMulgru_cell_57/Sigmoid_1:y:0gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul?
gru_cell_57/add_2AddV2gru_cell_57/split:output:2gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_2v
gru_cell_57/ReluRelugru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Relu?
gru_cell_57/mul_1Mulgru_cell_57/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_1k
gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_57/sub/x?
gru_cell_57/subSubgru_cell_57/sub/x:output:0gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/sub?
gru_cell_57/mul_2Mulgru_cell_57/sub:z:0gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_2?
gru_cell_57/add_3AddV2gru_cell_57/mul_1:z:0gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_57_readvariableop_resource*gru_cell_57_matmul_readvariableop_resource,gru_cell_57_matmul_1_readvariableop_resource*
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
bodyR
while_body_3598079*
condR
while_cond_3598078*9
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
NoOpNoOp"^gru_cell_57/MatMul/ReadVariableOp$^gru_cell_57/MatMul_1/ReadVariableOp^gru_cell_57/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_57/MatMul/ReadVariableOp!gru_cell_57/MatMul/ReadVariableOp2J
#gru_cell_57/MatMul_1/ReadVariableOp#gru_cell_57/MatMul_1/ReadVariableOp28
gru_cell_57/ReadVariableOpgru_cell_57/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?

J__inference_sequential_28_layer_call_and_return_conditional_losses_3599900

inputs=
*gru_56_gru_cell_56_readvariableop_resource:	?D
1gru_56_gru_cell_56_matmul_readvariableop_resource:	?G
3gru_56_gru_cell_56_matmul_1_readvariableop_resource:
??=
*gru_57_gru_cell_57_readvariableop_resource:	?E
1gru_57_gru_cell_57_matmul_readvariableop_resource:
??G
3gru_57_gru_cell_57_matmul_1_readvariableop_resource:
??>
*dense_63_tensordot_readvariableop_resource:
??7
(dense_63_biasadd_readvariableop_resource:	?>
*dense_64_tensordot_readvariableop_resource:
??7
(dense_64_biasadd_readvariableop_resource:	?=
*dense_65_tensordot_readvariableop_resource:	?6
(dense_65_biasadd_readvariableop_resource:
identity??dense_63/BiasAdd/ReadVariableOp?!dense_63/Tensordot/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?!dense_64/Tensordot/ReadVariableOp?dense_65/BiasAdd/ReadVariableOp?!dense_65/Tensordot/ReadVariableOp?(gru_56/gru_cell_56/MatMul/ReadVariableOp?*gru_56/gru_cell_56/MatMul_1/ReadVariableOp?!gru_56/gru_cell_56/ReadVariableOp?gru_56/while?(gru_57/gru_cell_57/MatMul/ReadVariableOp?*gru_57/gru_cell_57/MatMul_1/ReadVariableOp?!gru_57/gru_cell_57/ReadVariableOp?gru_57/whileR
gru_56/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_56/Shape?
gru_56/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_56/strided_slice/stack?
gru_56/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_56/strided_slice/stack_1?
gru_56/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_56/strided_slice/stack_2?
gru_56/strided_sliceStridedSlicegru_56/Shape:output:0#gru_56/strided_slice/stack:output:0%gru_56/strided_slice/stack_1:output:0%gru_56/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_56/strided_sliceq
gru_56/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_56/zeros/packed/1?
gru_56/zeros/packedPackgru_56/strided_slice:output:0gru_56/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_56/zeros/packedm
gru_56/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_56/zeros/Const?
gru_56/zerosFillgru_56/zeros/packed:output:0gru_56/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_56/zeros?
gru_56/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_56/transpose/perm?
gru_56/transpose	Transposeinputsgru_56/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
gru_56/transposed
gru_56/Shape_1Shapegru_56/transpose:y:0*
T0*
_output_shapes
:2
gru_56/Shape_1?
gru_56/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_56/strided_slice_1/stack?
gru_56/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_56/strided_slice_1/stack_1?
gru_56/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_56/strided_slice_1/stack_2?
gru_56/strided_slice_1StridedSlicegru_56/Shape_1:output:0%gru_56/strided_slice_1/stack:output:0'gru_56/strided_slice_1/stack_1:output:0'gru_56/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_56/strided_slice_1?
"gru_56/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_56/TensorArrayV2/element_shape?
gru_56/TensorArrayV2TensorListReserve+gru_56/TensorArrayV2/element_shape:output:0gru_56/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_56/TensorArrayV2?
<gru_56/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_56/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_56/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_56/transpose:y:0Egru_56/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_56/TensorArrayUnstack/TensorListFromTensor?
gru_56/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_56/strided_slice_2/stack?
gru_56/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_56/strided_slice_2/stack_1?
gru_56/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_56/strided_slice_2/stack_2?
gru_56/strided_slice_2StridedSlicegru_56/transpose:y:0%gru_56/strided_slice_2/stack:output:0'gru_56/strided_slice_2/stack_1:output:0'gru_56/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_56/strided_slice_2?
!gru_56/gru_cell_56/ReadVariableOpReadVariableOp*gru_56_gru_cell_56_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_56/gru_cell_56/ReadVariableOp?
gru_56/gru_cell_56/unstackUnpack)gru_56/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_56/gru_cell_56/unstack?
(gru_56/gru_cell_56/MatMul/ReadVariableOpReadVariableOp1gru_56_gru_cell_56_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_56/gru_cell_56/MatMul/ReadVariableOp?
gru_56/gru_cell_56/MatMulMatMulgru_56/strided_slice_2:output:00gru_56/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/MatMul?
gru_56/gru_cell_56/BiasAddBiasAdd#gru_56/gru_cell_56/MatMul:product:0#gru_56/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/BiasAdd?
"gru_56/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_56/gru_cell_56/split/split_dim?
gru_56/gru_cell_56/splitSplit+gru_56/gru_cell_56/split/split_dim:output:0#gru_56/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_56/gru_cell_56/split?
*gru_56/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp3gru_56_gru_cell_56_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_56/gru_cell_56/MatMul_1/ReadVariableOp?
gru_56/gru_cell_56/MatMul_1MatMulgru_56/zeros:output:02gru_56/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/MatMul_1?
gru_56/gru_cell_56/BiasAdd_1BiasAdd%gru_56/gru_cell_56/MatMul_1:product:0#gru_56/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/BiasAdd_1?
gru_56/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_56/gru_cell_56/Const?
$gru_56/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_56/gru_cell_56/split_1/split_dim?
gru_56/gru_cell_56/split_1SplitV%gru_56/gru_cell_56/BiasAdd_1:output:0!gru_56/gru_cell_56/Const:output:0-gru_56/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_56/gru_cell_56/split_1?
gru_56/gru_cell_56/addAddV2!gru_56/gru_cell_56/split:output:0#gru_56/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/add?
gru_56/gru_cell_56/SigmoidSigmoidgru_56/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/Sigmoid?
gru_56/gru_cell_56/add_1AddV2!gru_56/gru_cell_56/split:output:1#gru_56/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/add_1?
gru_56/gru_cell_56/Sigmoid_1Sigmoidgru_56/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/Sigmoid_1?
gru_56/gru_cell_56/mulMul gru_56/gru_cell_56/Sigmoid_1:y:0#gru_56/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/mul?
gru_56/gru_cell_56/add_2AddV2!gru_56/gru_cell_56/split:output:2gru_56/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/add_2?
gru_56/gru_cell_56/ReluRelugru_56/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/Relu?
gru_56/gru_cell_56/mul_1Mulgru_56/gru_cell_56/Sigmoid:y:0gru_56/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/mul_1y
gru_56/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_56/gru_cell_56/sub/x?
gru_56/gru_cell_56/subSub!gru_56/gru_cell_56/sub/x:output:0gru_56/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/sub?
gru_56/gru_cell_56/mul_2Mulgru_56/gru_cell_56/sub:z:0%gru_56/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/mul_2?
gru_56/gru_cell_56/add_3AddV2gru_56/gru_cell_56/mul_1:z:0gru_56/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/add_3?
$gru_56/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$gru_56/TensorArrayV2_1/element_shape?
gru_56/TensorArrayV2_1TensorListReserve-gru_56/TensorArrayV2_1/element_shape:output:0gru_56/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_56/TensorArrayV2_1\
gru_56/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_56/time?
gru_56/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_56/while/maximum_iterationsx
gru_56/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_56/while/loop_counter?
gru_56/whileWhile"gru_56/while/loop_counter:output:0(gru_56/while/maximum_iterations:output:0gru_56/time:output:0gru_56/TensorArrayV2_1:handle:0gru_56/zeros:output:0gru_56/strided_slice_1:output:0>gru_56/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_56_gru_cell_56_readvariableop_resource1gru_56_gru_cell_56_matmul_readvariableop_resource3gru_56_gru_cell_56_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *%
bodyR
gru_56_while_body_3599550*%
condR
gru_56_while_cond_3599549*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_56/while?
7gru_56/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7gru_56/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_56/TensorArrayV2Stack/TensorListStackTensorListStackgru_56/while:output:3@gru_56/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_56/TensorArrayV2Stack/TensorListStack?
gru_56/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_56/strided_slice_3/stack?
gru_56/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_56/strided_slice_3/stack_1?
gru_56/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_56/strided_slice_3/stack_2?
gru_56/strided_slice_3StridedSlice2gru_56/TensorArrayV2Stack/TensorListStack:tensor:0%gru_56/strided_slice_3/stack:output:0'gru_56/strided_slice_3/stack_1:output:0'gru_56/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_56/strided_slice_3?
gru_56/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_56/transpose_1/perm?
gru_56/transpose_1	Transpose2gru_56/TensorArrayV2Stack/TensorListStack:tensor:0 gru_56/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_56/transpose_1t
gru_56/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_56/runtimey
dropout_91/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_91/dropout/Const?
dropout_91/dropout/MulMulgru_56/transpose_1:y:0!dropout_91/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_91/dropout/Mulz
dropout_91/dropout/ShapeShapegru_56/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_91/dropout/Shape?
/dropout_91/dropout/random_uniform/RandomUniformRandomUniform!dropout_91/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_91/dropout/random_uniform/RandomUniform?
!dropout_91/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_91/dropout/GreaterEqual/y?
dropout_91/dropout/GreaterEqualGreaterEqual8dropout_91/dropout/random_uniform/RandomUniform:output:0*dropout_91/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_91/dropout/GreaterEqual?
dropout_91/dropout/CastCast#dropout_91/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_91/dropout/Cast?
dropout_91/dropout/Mul_1Muldropout_91/dropout/Mul:z:0dropout_91/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_91/dropout/Mul_1h
gru_57/ShapeShapedropout_91/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
gru_57/Shape?
gru_57/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_57/strided_slice/stack?
gru_57/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_57/strided_slice/stack_1?
gru_57/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_57/strided_slice/stack_2?
gru_57/strided_sliceStridedSlicegru_57/Shape:output:0#gru_57/strided_slice/stack:output:0%gru_57/strided_slice/stack_1:output:0%gru_57/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_57/strided_sliceq
gru_57/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_57/zeros/packed/1?
gru_57/zeros/packedPackgru_57/strided_slice:output:0gru_57/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_57/zeros/packedm
gru_57/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_57/zeros/Const?
gru_57/zerosFillgru_57/zeros/packed:output:0gru_57/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_57/zeros?
gru_57/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_57/transpose/perm?
gru_57/transpose	Transposedropout_91/dropout/Mul_1:z:0gru_57/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_57/transposed
gru_57/Shape_1Shapegru_57/transpose:y:0*
T0*
_output_shapes
:2
gru_57/Shape_1?
gru_57/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_57/strided_slice_1/stack?
gru_57/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_57/strided_slice_1/stack_1?
gru_57/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_57/strided_slice_1/stack_2?
gru_57/strided_slice_1StridedSlicegru_57/Shape_1:output:0%gru_57/strided_slice_1/stack:output:0'gru_57/strided_slice_1/stack_1:output:0'gru_57/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_57/strided_slice_1?
"gru_57/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_57/TensorArrayV2/element_shape?
gru_57/TensorArrayV2TensorListReserve+gru_57/TensorArrayV2/element_shape:output:0gru_57/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_57/TensorArrayV2?
<gru_57/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_57/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_57/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_57/transpose:y:0Egru_57/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_57/TensorArrayUnstack/TensorListFromTensor?
gru_57/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_57/strided_slice_2/stack?
gru_57/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_57/strided_slice_2/stack_1?
gru_57/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_57/strided_slice_2/stack_2?
gru_57/strided_slice_2StridedSlicegru_57/transpose:y:0%gru_57/strided_slice_2/stack:output:0'gru_57/strided_slice_2/stack_1:output:0'gru_57/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_57/strided_slice_2?
!gru_57/gru_cell_57/ReadVariableOpReadVariableOp*gru_57_gru_cell_57_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_57/gru_cell_57/ReadVariableOp?
gru_57/gru_cell_57/unstackUnpack)gru_57/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_57/gru_cell_57/unstack?
(gru_57/gru_cell_57/MatMul/ReadVariableOpReadVariableOp1gru_57_gru_cell_57_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(gru_57/gru_cell_57/MatMul/ReadVariableOp?
gru_57/gru_cell_57/MatMulMatMulgru_57/strided_slice_2:output:00gru_57/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/MatMul?
gru_57/gru_cell_57/BiasAddBiasAdd#gru_57/gru_cell_57/MatMul:product:0#gru_57/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/BiasAdd?
"gru_57/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_57/gru_cell_57/split/split_dim?
gru_57/gru_cell_57/splitSplit+gru_57/gru_cell_57/split/split_dim:output:0#gru_57/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_57/gru_cell_57/split?
*gru_57/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp3gru_57_gru_cell_57_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_57/gru_cell_57/MatMul_1/ReadVariableOp?
gru_57/gru_cell_57/MatMul_1MatMulgru_57/zeros:output:02gru_57/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/MatMul_1?
gru_57/gru_cell_57/BiasAdd_1BiasAdd%gru_57/gru_cell_57/MatMul_1:product:0#gru_57/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/BiasAdd_1?
gru_57/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_57/gru_cell_57/Const?
$gru_57/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_57/gru_cell_57/split_1/split_dim?
gru_57/gru_cell_57/split_1SplitV%gru_57/gru_cell_57/BiasAdd_1:output:0!gru_57/gru_cell_57/Const:output:0-gru_57/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_57/gru_cell_57/split_1?
gru_57/gru_cell_57/addAddV2!gru_57/gru_cell_57/split:output:0#gru_57/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/add?
gru_57/gru_cell_57/SigmoidSigmoidgru_57/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/Sigmoid?
gru_57/gru_cell_57/add_1AddV2!gru_57/gru_cell_57/split:output:1#gru_57/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/add_1?
gru_57/gru_cell_57/Sigmoid_1Sigmoidgru_57/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/Sigmoid_1?
gru_57/gru_cell_57/mulMul gru_57/gru_cell_57/Sigmoid_1:y:0#gru_57/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/mul?
gru_57/gru_cell_57/add_2AddV2!gru_57/gru_cell_57/split:output:2gru_57/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/add_2?
gru_57/gru_cell_57/ReluRelugru_57/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/Relu?
gru_57/gru_cell_57/mul_1Mulgru_57/gru_cell_57/Sigmoid:y:0gru_57/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/mul_1y
gru_57/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_57/gru_cell_57/sub/x?
gru_57/gru_cell_57/subSub!gru_57/gru_cell_57/sub/x:output:0gru_57/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/sub?
gru_57/gru_cell_57/mul_2Mulgru_57/gru_cell_57/sub:z:0%gru_57/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/mul_2?
gru_57/gru_cell_57/add_3AddV2gru_57/gru_cell_57/mul_1:z:0gru_57/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/add_3?
$gru_57/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$gru_57/TensorArrayV2_1/element_shape?
gru_57/TensorArrayV2_1TensorListReserve-gru_57/TensorArrayV2_1/element_shape:output:0gru_57/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_57/TensorArrayV2_1\
gru_57/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_57/time?
gru_57/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_57/while/maximum_iterationsx
gru_57/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_57/while/loop_counter?
gru_57/whileWhile"gru_57/while/loop_counter:output:0(gru_57/while/maximum_iterations:output:0gru_57/time:output:0gru_57/TensorArrayV2_1:handle:0gru_57/zeros:output:0gru_57/strided_slice_1:output:0>gru_57/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_57_gru_cell_57_readvariableop_resource1gru_57_gru_cell_57_matmul_readvariableop_resource3gru_57_gru_cell_57_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *%
bodyR
gru_57_while_body_3599707*%
condR
gru_57_while_cond_3599706*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_57/while?
7gru_57/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7gru_57/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_57/TensorArrayV2Stack/TensorListStackTensorListStackgru_57/while:output:3@gru_57/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_57/TensorArrayV2Stack/TensorListStack?
gru_57/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_57/strided_slice_3/stack?
gru_57/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_57/strided_slice_3/stack_1?
gru_57/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_57/strided_slice_3/stack_2?
gru_57/strided_slice_3StridedSlice2gru_57/TensorArrayV2Stack/TensorListStack:tensor:0%gru_57/strided_slice_3/stack:output:0'gru_57/strided_slice_3/stack_1:output:0'gru_57/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_57/strided_slice_3?
gru_57/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_57/transpose_1/perm?
gru_57/transpose_1	Transpose2gru_57/TensorArrayV2Stack/TensorListStack:tensor:0 gru_57/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_57/transpose_1t
gru_57/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_57/runtimey
dropout_92/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_92/dropout/Const?
dropout_92/dropout/MulMulgru_57/transpose_1:y:0!dropout_92/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_92/dropout/Mulz
dropout_92/dropout/ShapeShapegru_57/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_92/dropout/Shape?
/dropout_92/dropout/random_uniform/RandomUniformRandomUniform!dropout_92/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_92/dropout/random_uniform/RandomUniform?
!dropout_92/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_92/dropout/GreaterEqual/y?
dropout_92/dropout/GreaterEqualGreaterEqual8dropout_92/dropout/random_uniform/RandomUniform:output:0*dropout_92/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_92/dropout/GreaterEqual?
dropout_92/dropout/CastCast#dropout_92/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_92/dropout/Cast?
dropout_92/dropout/Mul_1Muldropout_92/dropout/Mul:z:0dropout_92/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_92/dropout/Mul_1?
!dense_63/Tensordot/ReadVariableOpReadVariableOp*dense_63_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_63/Tensordot/ReadVariableOp|
dense_63/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_63/Tensordot/axes?
dense_63/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_63/Tensordot/free?
dense_63/Tensordot/ShapeShapedropout_92/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_63/Tensordot/Shape?
 dense_63/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_63/Tensordot/GatherV2/axis?
dense_63/Tensordot/GatherV2GatherV2!dense_63/Tensordot/Shape:output:0 dense_63/Tensordot/free:output:0)dense_63/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_63/Tensordot/GatherV2?
"dense_63/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_63/Tensordot/GatherV2_1/axis?
dense_63/Tensordot/GatherV2_1GatherV2!dense_63/Tensordot/Shape:output:0 dense_63/Tensordot/axes:output:0+dense_63/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_63/Tensordot/GatherV2_1~
dense_63/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_63/Tensordot/Const?
dense_63/Tensordot/ProdProd$dense_63/Tensordot/GatherV2:output:0!dense_63/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_63/Tensordot/Prod?
dense_63/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_63/Tensordot/Const_1?
dense_63/Tensordot/Prod_1Prod&dense_63/Tensordot/GatherV2_1:output:0#dense_63/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_63/Tensordot/Prod_1?
dense_63/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_63/Tensordot/concat/axis?
dense_63/Tensordot/concatConcatV2 dense_63/Tensordot/free:output:0 dense_63/Tensordot/axes:output:0'dense_63/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_63/Tensordot/concat?
dense_63/Tensordot/stackPack dense_63/Tensordot/Prod:output:0"dense_63/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_63/Tensordot/stack?
dense_63/Tensordot/transpose	Transposedropout_92/dropout/Mul_1:z:0"dense_63/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_63/Tensordot/transpose?
dense_63/Tensordot/ReshapeReshape dense_63/Tensordot/transpose:y:0!dense_63/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_63/Tensordot/Reshape?
dense_63/Tensordot/MatMulMatMul#dense_63/Tensordot/Reshape:output:0)dense_63/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_63/Tensordot/MatMul?
dense_63/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_63/Tensordot/Const_2?
 dense_63/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_63/Tensordot/concat_1/axis?
dense_63/Tensordot/concat_1ConcatV2$dense_63/Tensordot/GatherV2:output:0#dense_63/Tensordot/Const_2:output:0)dense_63/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_63/Tensordot/concat_1?
dense_63/TensordotReshape#dense_63/Tensordot/MatMul:product:0$dense_63/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_63/Tensordot?
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_63/BiasAdd/ReadVariableOp?
dense_63/BiasAddBiasAdddense_63/Tensordot:output:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_63/BiasAddx
dense_63/ReluReludense_63/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_63/Reluy
dropout_93/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_93/dropout/Const?
dropout_93/dropout/MulMuldense_63/Relu:activations:0!dropout_93/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_93/dropout/Mul
dropout_93/dropout/ShapeShapedense_63/Relu:activations:0*
T0*
_output_shapes
:2
dropout_93/dropout/Shape?
/dropout_93/dropout/random_uniform/RandomUniformRandomUniform!dropout_93/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_93/dropout/random_uniform/RandomUniform?
!dropout_93/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_93/dropout/GreaterEqual/y?
dropout_93/dropout/GreaterEqualGreaterEqual8dropout_93/dropout/random_uniform/RandomUniform:output:0*dropout_93/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_93/dropout/GreaterEqual?
dropout_93/dropout/CastCast#dropout_93/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_93/dropout/Cast?
dropout_93/dropout/Mul_1Muldropout_93/dropout/Mul:z:0dropout_93/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_93/dropout/Mul_1?
!dense_64/Tensordot/ReadVariableOpReadVariableOp*dense_64_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_64/Tensordot/ReadVariableOp|
dense_64/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_64/Tensordot/axes?
dense_64/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_64/Tensordot/free?
dense_64/Tensordot/ShapeShapedropout_93/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_64/Tensordot/Shape?
 dense_64/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_64/Tensordot/GatherV2/axis?
dense_64/Tensordot/GatherV2GatherV2!dense_64/Tensordot/Shape:output:0 dense_64/Tensordot/free:output:0)dense_64/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_64/Tensordot/GatherV2?
"dense_64/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_64/Tensordot/GatherV2_1/axis?
dense_64/Tensordot/GatherV2_1GatherV2!dense_64/Tensordot/Shape:output:0 dense_64/Tensordot/axes:output:0+dense_64/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_64/Tensordot/GatherV2_1~
dense_64/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_64/Tensordot/Const?
dense_64/Tensordot/ProdProd$dense_64/Tensordot/GatherV2:output:0!dense_64/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_64/Tensordot/Prod?
dense_64/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_64/Tensordot/Const_1?
dense_64/Tensordot/Prod_1Prod&dense_64/Tensordot/GatherV2_1:output:0#dense_64/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_64/Tensordot/Prod_1?
dense_64/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_64/Tensordot/concat/axis?
dense_64/Tensordot/concatConcatV2 dense_64/Tensordot/free:output:0 dense_64/Tensordot/axes:output:0'dense_64/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_64/Tensordot/concat?
dense_64/Tensordot/stackPack dense_64/Tensordot/Prod:output:0"dense_64/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_64/Tensordot/stack?
dense_64/Tensordot/transpose	Transposedropout_93/dropout/Mul_1:z:0"dense_64/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_64/Tensordot/transpose?
dense_64/Tensordot/ReshapeReshape dense_64/Tensordot/transpose:y:0!dense_64/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_64/Tensordot/Reshape?
dense_64/Tensordot/MatMulMatMul#dense_64/Tensordot/Reshape:output:0)dense_64/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_64/Tensordot/MatMul?
dense_64/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_64/Tensordot/Const_2?
 dense_64/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_64/Tensordot/concat_1/axis?
dense_64/Tensordot/concat_1ConcatV2$dense_64/Tensordot/GatherV2:output:0#dense_64/Tensordot/Const_2:output:0)dense_64/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_64/Tensordot/concat_1?
dense_64/TensordotReshape#dense_64/Tensordot/MatMul:product:0$dense_64/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_64/Tensordot?
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_64/BiasAdd/ReadVariableOp?
dense_64/BiasAddBiasAdddense_64/Tensordot:output:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_64/BiasAddx
dense_64/ReluReludense_64/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_64/Reluy
dropout_94/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_94/dropout/Const?
dropout_94/dropout/MulMuldense_64/Relu:activations:0!dropout_94/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_94/dropout/Mul
dropout_94/dropout/ShapeShapedense_64/Relu:activations:0*
T0*
_output_shapes
:2
dropout_94/dropout/Shape?
/dropout_94/dropout/random_uniform/RandomUniformRandomUniform!dropout_94/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_94/dropout/random_uniform/RandomUniform?
!dropout_94/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_94/dropout/GreaterEqual/y?
dropout_94/dropout/GreaterEqualGreaterEqual8dropout_94/dropout/random_uniform/RandomUniform:output:0*dropout_94/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_94/dropout/GreaterEqual?
dropout_94/dropout/CastCast#dropout_94/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_94/dropout/Cast?
dropout_94/dropout/Mul_1Muldropout_94/dropout/Mul:z:0dropout_94/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_94/dropout/Mul_1?
!dense_65/Tensordot/ReadVariableOpReadVariableOp*dense_65_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_65/Tensordot/ReadVariableOp|
dense_65/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_65/Tensordot/axes?
dense_65/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_65/Tensordot/free?
dense_65/Tensordot/ShapeShapedropout_94/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_65/Tensordot/Shape?
 dense_65/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_65/Tensordot/GatherV2/axis?
dense_65/Tensordot/GatherV2GatherV2!dense_65/Tensordot/Shape:output:0 dense_65/Tensordot/free:output:0)dense_65/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_65/Tensordot/GatherV2?
"dense_65/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_65/Tensordot/GatherV2_1/axis?
dense_65/Tensordot/GatherV2_1GatherV2!dense_65/Tensordot/Shape:output:0 dense_65/Tensordot/axes:output:0+dense_65/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_65/Tensordot/GatherV2_1~
dense_65/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_65/Tensordot/Const?
dense_65/Tensordot/ProdProd$dense_65/Tensordot/GatherV2:output:0!dense_65/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_65/Tensordot/Prod?
dense_65/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_65/Tensordot/Const_1?
dense_65/Tensordot/Prod_1Prod&dense_65/Tensordot/GatherV2_1:output:0#dense_65/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_65/Tensordot/Prod_1?
dense_65/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_65/Tensordot/concat/axis?
dense_65/Tensordot/concatConcatV2 dense_65/Tensordot/free:output:0 dense_65/Tensordot/axes:output:0'dense_65/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_65/Tensordot/concat?
dense_65/Tensordot/stackPack dense_65/Tensordot/Prod:output:0"dense_65/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_65/Tensordot/stack?
dense_65/Tensordot/transpose	Transposedropout_94/dropout/Mul_1:z:0"dense_65/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_65/Tensordot/transpose?
dense_65/Tensordot/ReshapeReshape dense_65/Tensordot/transpose:y:0!dense_65/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_65/Tensordot/Reshape?
dense_65/Tensordot/MatMulMatMul#dense_65/Tensordot/Reshape:output:0)dense_65/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_65/Tensordot/MatMul?
dense_65/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_65/Tensordot/Const_2?
 dense_65/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_65/Tensordot/concat_1/axis?
dense_65/Tensordot/concat_1ConcatV2$dense_65/Tensordot/GatherV2:output:0#dense_65/Tensordot/Const_2:output:0)dense_65/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_65/Tensordot/concat_1?
dense_65/TensordotReshape#dense_65/Tensordot/MatMul:product:0$dense_65/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_65/Tensordot?
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_65/BiasAdd/ReadVariableOp?
dense_65/BiasAddBiasAdddense_65/Tensordot:output:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_65/BiasAddx
IdentityIdentitydense_65/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_63/BiasAdd/ReadVariableOp"^dense_63/Tensordot/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp"^dense_64/Tensordot/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp"^dense_65/Tensordot/ReadVariableOp)^gru_56/gru_cell_56/MatMul/ReadVariableOp+^gru_56/gru_cell_56/MatMul_1/ReadVariableOp"^gru_56/gru_cell_56/ReadVariableOp^gru_56/while)^gru_57/gru_cell_57/MatMul/ReadVariableOp+^gru_57/gru_cell_57/MatMul_1/ReadVariableOp"^gru_57/gru_cell_57/ReadVariableOp^gru_57/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2F
!dense_63/Tensordot/ReadVariableOp!dense_63/Tensordot/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2F
!dense_64/Tensordot/ReadVariableOp!dense_64/Tensordot/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2F
!dense_65/Tensordot/ReadVariableOp!dense_65/Tensordot/ReadVariableOp2T
(gru_56/gru_cell_56/MatMul/ReadVariableOp(gru_56/gru_cell_56/MatMul/ReadVariableOp2X
*gru_56/gru_cell_56/MatMul_1/ReadVariableOp*gru_56/gru_cell_56/MatMul_1/ReadVariableOp2F
!gru_56/gru_cell_56/ReadVariableOp!gru_56/gru_cell_56/ReadVariableOp2
gru_56/whilegru_56/while2T
(gru_57/gru_cell_57/MatMul/ReadVariableOp(gru_57/gru_cell_57/MatMul/ReadVariableOp2X
*gru_57/gru_cell_57/MatMul_1/ReadVariableOp*gru_57/gru_cell_57/MatMul_1/ReadVariableOp2F
!gru_57/gru_cell_57/ReadVariableOp!gru_57/gru_cell_57/ReadVariableOp2
gru_57/whilegru_57/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_93_layer_call_fn_3601316

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
GPU 2J 8? *P
fKRI
G__inference_dropout_93_layer_call_and_return_conditional_losses_35983982
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
?X
?
C__inference_gru_57_layer_call_and_return_conditional_losses_3601239

inputs6
#gru_cell_57_readvariableop_resource:	?>
*gru_cell_57_matmul_readvariableop_resource:
??@
,gru_cell_57_matmul_1_readvariableop_resource:
??
identity??!gru_cell_57/MatMul/ReadVariableOp?#gru_cell_57/MatMul_1/ReadVariableOp?gru_cell_57/ReadVariableOp?whileD
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
gru_cell_57/ReadVariableOpReadVariableOp#gru_cell_57_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_57/ReadVariableOp?
gru_cell_57/unstackUnpack"gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_57/unstack?
!gru_cell_57/MatMul/ReadVariableOpReadVariableOp*gru_cell_57_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_57/MatMul/ReadVariableOp?
gru_cell_57/MatMulMatMulstrided_slice_2:output:0)gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul?
gru_cell_57/BiasAddBiasAddgru_cell_57/MatMul:product:0gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd?
gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split/split_dim?
gru_cell_57/splitSplit$gru_cell_57/split/split_dim:output:0gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split?
#gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_57_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_57/MatMul_1/ReadVariableOp?
gru_cell_57/MatMul_1MatMulzeros:output:0+gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul_1?
gru_cell_57/BiasAdd_1BiasAddgru_cell_57/MatMul_1:product:0gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd_1{
gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_57/Const?
gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split_1/split_dim?
gru_cell_57/split_1SplitVgru_cell_57/BiasAdd_1:output:0gru_cell_57/Const:output:0&gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split_1?
gru_cell_57/addAddV2gru_cell_57/split:output:0gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add}
gru_cell_57/SigmoidSigmoidgru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid?
gru_cell_57/add_1AddV2gru_cell_57/split:output:1gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_1?
gru_cell_57/Sigmoid_1Sigmoidgru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid_1?
gru_cell_57/mulMulgru_cell_57/Sigmoid_1:y:0gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul?
gru_cell_57/add_2AddV2gru_cell_57/split:output:2gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_2v
gru_cell_57/ReluRelugru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Relu?
gru_cell_57/mul_1Mulgru_cell_57/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_1k
gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_57/sub/x?
gru_cell_57/subSubgru_cell_57/sub/x:output:0gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/sub?
gru_cell_57/mul_2Mulgru_cell_57/sub:z:0gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_2?
gru_cell_57/add_3AddV2gru_cell_57/mul_1:z:0gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_57_readvariableop_resource*gru_cell_57_matmul_readvariableop_resource,gru_cell_57_matmul_1_readvariableop_resource*
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
bodyR
while_body_3601150*
condR
while_cond_3601149*9
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
NoOpNoOp"^gru_cell_57/MatMul/ReadVariableOp$^gru_cell_57/MatMul_1/ReadVariableOp^gru_cell_57/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_57/MatMul/ReadVariableOp!gru_cell_57/MatMul/ReadVariableOp2J
#gru_cell_57/MatMul_1/ReadVariableOp#gru_cell_57/MatMul_1/ReadVariableOp28
gru_cell_57/ReadVariableOpgru_cell_57/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_3600466
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3600466___redundant_placeholder05
1while_while_cond_3600466___redundant_placeholder15
1while_while_cond_3600466___redundant_placeholder25
1while_while_cond_3600466___redundant_placeholder3
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
?
/__inference_sequential_28_layer_call_fn_3599100

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
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_28_layer_call_and_return_conditional_losses_35988752
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
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?Y
?
C__inference_gru_57_layer_call_and_return_conditional_losses_3600933
inputs_06
#gru_cell_57_readvariableop_resource:	?>
*gru_cell_57_matmul_readvariableop_resource:
??@
,gru_cell_57_matmul_1_readvariableop_resource:
??
identity??!gru_cell_57/MatMul/ReadVariableOp?#gru_cell_57/MatMul_1/ReadVariableOp?gru_cell_57/ReadVariableOp?whileF
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
gru_cell_57/ReadVariableOpReadVariableOp#gru_cell_57_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_57/ReadVariableOp?
gru_cell_57/unstackUnpack"gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_57/unstack?
!gru_cell_57/MatMul/ReadVariableOpReadVariableOp*gru_cell_57_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_57/MatMul/ReadVariableOp?
gru_cell_57/MatMulMatMulstrided_slice_2:output:0)gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul?
gru_cell_57/BiasAddBiasAddgru_cell_57/MatMul:product:0gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd?
gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split/split_dim?
gru_cell_57/splitSplit$gru_cell_57/split/split_dim:output:0gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split?
#gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_57_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_57/MatMul_1/ReadVariableOp?
gru_cell_57/MatMul_1MatMulzeros:output:0+gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul_1?
gru_cell_57/BiasAdd_1BiasAddgru_cell_57/MatMul_1:product:0gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd_1{
gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_57/Const?
gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split_1/split_dim?
gru_cell_57/split_1SplitVgru_cell_57/BiasAdd_1:output:0gru_cell_57/Const:output:0&gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split_1?
gru_cell_57/addAddV2gru_cell_57/split:output:0gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add}
gru_cell_57/SigmoidSigmoidgru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid?
gru_cell_57/add_1AddV2gru_cell_57/split:output:1gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_1?
gru_cell_57/Sigmoid_1Sigmoidgru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid_1?
gru_cell_57/mulMulgru_cell_57/Sigmoid_1:y:0gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul?
gru_cell_57/add_2AddV2gru_cell_57/split:output:2gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_2v
gru_cell_57/ReluRelugru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Relu?
gru_cell_57/mul_1Mulgru_cell_57/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_1k
gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_57/sub/x?
gru_cell_57/subSubgru_cell_57/sub/x:output:0gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/sub?
gru_cell_57/mul_2Mulgru_cell_57/sub:z:0gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_2?
gru_cell_57/add_3AddV2gru_cell_57/mul_1:z:0gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_57_readvariableop_resource*gru_cell_57_matmul_readvariableop_resource,gru_cell_57_matmul_1_readvariableop_resource*
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
bodyR
while_body_3600844*
condR
while_cond_3600843*9
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
NoOpNoOp"^gru_cell_57/MatMul/ReadVariableOp$^gru_cell_57/MatMul_1/ReadVariableOp^gru_cell_57/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2F
!gru_cell_57/MatMul/ReadVariableOp!gru_cell_57/MatMul/ReadVariableOp2J
#gru_cell_57/MatMul_1/ReadVariableOp#gru_cell_57/MatMul_1/ReadVariableOp28
gru_cell_57/ReadVariableOpgru_cell_57/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
e
G__inference_dropout_92_layer_call_and_return_conditional_losses_3598181

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
?E
?
while_body_3600691
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_57_readvariableop_resource_0:	?F
2while_gru_cell_57_matmul_readvariableop_resource_0:
??H
4while_gru_cell_57_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_57_readvariableop_resource:	?D
0while_gru_cell_57_matmul_readvariableop_resource:
??F
2while_gru_cell_57_matmul_1_readvariableop_resource:
????'while/gru_cell_57/MatMul/ReadVariableOp?)while/gru_cell_57/MatMul_1/ReadVariableOp? while/gru_cell_57/ReadVariableOp?
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
 while/gru_cell_57/ReadVariableOpReadVariableOp+while_gru_cell_57_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_57/ReadVariableOp?
while/gru_cell_57/unstackUnpack(while/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_57/unstack?
'while/gru_cell_57/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_57_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_57/MatMul/ReadVariableOp?
while/gru_cell_57/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul?
while/gru_cell_57/BiasAddBiasAdd"while/gru_cell_57/MatMul:product:0"while/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd?
!while/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_57/split/split_dim?
while/gru_cell_57/splitSplit*while/gru_cell_57/split/split_dim:output:0"while/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split?
)while/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_57_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_57/MatMul_1/ReadVariableOp?
while/gru_cell_57/MatMul_1MatMulwhile_placeholder_21while/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul_1?
while/gru_cell_57/BiasAdd_1BiasAdd$while/gru_cell_57/MatMul_1:product:0"while/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd_1?
while/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_57/Const?
#while/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_57/split_1/split_dim?
while/gru_cell_57/split_1SplitV$while/gru_cell_57/BiasAdd_1:output:0 while/gru_cell_57/Const:output:0,while/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split_1?
while/gru_cell_57/addAddV2 while/gru_cell_57/split:output:0"while/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add?
while/gru_cell_57/SigmoidSigmoidwhile/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid?
while/gru_cell_57/add_1AddV2 while/gru_cell_57/split:output:1"while/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_1?
while/gru_cell_57/Sigmoid_1Sigmoidwhile/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid_1?
while/gru_cell_57/mulMulwhile/gru_cell_57/Sigmoid_1:y:0"while/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul?
while/gru_cell_57/add_2AddV2 while/gru_cell_57/split:output:2while/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_2?
while/gru_cell_57/ReluReluwhile/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Relu?
while/gru_cell_57/mul_1Mulwhile/gru_cell_57/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_1w
while/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_57/sub/x?
while/gru_cell_57/subSub while/gru_cell_57/sub/x:output:0while/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/sub?
while/gru_cell_57/mul_2Mulwhile/gru_cell_57/sub:z:0$while/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_2?
while/gru_cell_57/add_3AddV2while/gru_cell_57/mul_1:z:0while/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_57/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_57/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_57/MatMul/ReadVariableOp*^while/gru_cell_57/MatMul_1/ReadVariableOp!^while/gru_cell_57/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_57_matmul_1_readvariableop_resource4while_gru_cell_57_matmul_1_readvariableop_resource_0"f
0while_gru_cell_57_matmul_readvariableop_resource2while_gru_cell_57_matmul_readvariableop_resource_0"X
)while_gru_cell_57_readvariableop_resource+while_gru_cell_57_readvariableop_resource_0")
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
'while/gru_cell_57/MatMul/ReadVariableOp'while/gru_cell_57/MatMul/ReadVariableOp2V
)while/gru_cell_57/MatMul_1/ReadVariableOp)while/gru_cell_57/MatMul_1/ReadVariableOp2D
 while/gru_cell_57/ReadVariableOp while/gru_cell_57/ReadVariableOp: 
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
?
/__inference_sequential_28_layer_call_fn_3599071

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
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_28_layer_call_and_return_conditional_losses_35983082
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
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_28_layer_call_fn_3598335
gru_56_input
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
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_56_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_28_layer_call_and_return_conditional_losses_35983082
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
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_56_input
?!
?
E__inference_dense_63_layer_call_and_return_conditional_losses_3598214

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
?E
?
while_body_3600844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_57_readvariableop_resource_0:	?F
2while_gru_cell_57_matmul_readvariableop_resource_0:
??H
4while_gru_cell_57_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_57_readvariableop_resource:	?D
0while_gru_cell_57_matmul_readvariableop_resource:
??F
2while_gru_cell_57_matmul_1_readvariableop_resource:
????'while/gru_cell_57/MatMul/ReadVariableOp?)while/gru_cell_57/MatMul_1/ReadVariableOp? while/gru_cell_57/ReadVariableOp?
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
 while/gru_cell_57/ReadVariableOpReadVariableOp+while_gru_cell_57_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_57/ReadVariableOp?
while/gru_cell_57/unstackUnpack(while/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_57/unstack?
'while/gru_cell_57/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_57_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_57/MatMul/ReadVariableOp?
while/gru_cell_57/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul?
while/gru_cell_57/BiasAddBiasAdd"while/gru_cell_57/MatMul:product:0"while/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd?
!while/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_57/split/split_dim?
while/gru_cell_57/splitSplit*while/gru_cell_57/split/split_dim:output:0"while/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split?
)while/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_57_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_57/MatMul_1/ReadVariableOp?
while/gru_cell_57/MatMul_1MatMulwhile_placeholder_21while/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul_1?
while/gru_cell_57/BiasAdd_1BiasAdd$while/gru_cell_57/MatMul_1:product:0"while/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd_1?
while/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_57/Const?
#while/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_57/split_1/split_dim?
while/gru_cell_57/split_1SplitV$while/gru_cell_57/BiasAdd_1:output:0 while/gru_cell_57/Const:output:0,while/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split_1?
while/gru_cell_57/addAddV2 while/gru_cell_57/split:output:0"while/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add?
while/gru_cell_57/SigmoidSigmoidwhile/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid?
while/gru_cell_57/add_1AddV2 while/gru_cell_57/split:output:1"while/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_1?
while/gru_cell_57/Sigmoid_1Sigmoidwhile/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid_1?
while/gru_cell_57/mulMulwhile/gru_cell_57/Sigmoid_1:y:0"while/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul?
while/gru_cell_57/add_2AddV2 while/gru_cell_57/split:output:2while/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_2?
while/gru_cell_57/ReluReluwhile/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Relu?
while/gru_cell_57/mul_1Mulwhile/gru_cell_57/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_1w
while/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_57/sub/x?
while/gru_cell_57/subSub while/gru_cell_57/sub/x:output:0while/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/sub?
while/gru_cell_57/mul_2Mulwhile/gru_cell_57/sub:z:0$while/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_2?
while/gru_cell_57/add_3AddV2while/gru_cell_57/mul_1:z:0while/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_57/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_57/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_57/MatMul/ReadVariableOp*^while/gru_cell_57/MatMul_1/ReadVariableOp!^while/gru_cell_57/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_57_matmul_1_readvariableop_resource4while_gru_cell_57_matmul_1_readvariableop_resource_0"f
0while_gru_cell_57_matmul_readvariableop_resource2while_gru_cell_57_matmul_readvariableop_resource_0"X
)while_gru_cell_57_readvariableop_resource+while_gru_cell_57_readvariableop_resource_0")
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
'while/gru_cell_57/MatMul/ReadVariableOp'while/gru_cell_57/MatMul/ReadVariableOp2V
)while/gru_cell_57/MatMul_1/ReadVariableOp)while/gru_cell_57/MatMul_1/ReadVariableOp2D
 while/gru_cell_57/ReadVariableOp while/gru_cell_57/ReadVariableOp: 
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
?
while_cond_3597911
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3597911___redundant_placeholder05
1while_while_cond_3597911___redundant_placeholder15
1while_while_cond_3597911___redundant_placeholder25
1while_while_cond_3597911___redundant_placeholder3
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
?;
?
C__inference_gru_56_layer_call_and_return_conditional_losses_3597050

inputs&
gru_cell_56_3596974:	?&
gru_cell_56_3596976:	?'
gru_cell_56_3596978:
??
identity??#gru_cell_56/StatefulPartitionedCall?whileD
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
#gru_cell_56/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_56_3596974gru_cell_56_3596976gru_cell_56_3596978*
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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_35969232%
#gru_cell_56/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_56_3596974gru_cell_56_3596976gru_cell_56_3596978*
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
bodyR
while_body_3596986*
condR
while_cond_3596985*9
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
NoOpNoOp$^gru_cell_56/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#gru_cell_56/StatefulPartitionedCall#gru_cell_56/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?P
?	
gru_56_while_body_3599550*
&gru_56_while_gru_56_while_loop_counter0
,gru_56_while_gru_56_while_maximum_iterations
gru_56_while_placeholder
gru_56_while_placeholder_1
gru_56_while_placeholder_2)
%gru_56_while_gru_56_strided_slice_1_0e
agru_56_while_tensorarrayv2read_tensorlistgetitem_gru_56_tensorarrayunstack_tensorlistfromtensor_0E
2gru_56_while_gru_cell_56_readvariableop_resource_0:	?L
9gru_56_while_gru_cell_56_matmul_readvariableop_resource_0:	?O
;gru_56_while_gru_cell_56_matmul_1_readvariableop_resource_0:
??
gru_56_while_identity
gru_56_while_identity_1
gru_56_while_identity_2
gru_56_while_identity_3
gru_56_while_identity_4'
#gru_56_while_gru_56_strided_slice_1c
_gru_56_while_tensorarrayv2read_tensorlistgetitem_gru_56_tensorarrayunstack_tensorlistfromtensorC
0gru_56_while_gru_cell_56_readvariableop_resource:	?J
7gru_56_while_gru_cell_56_matmul_readvariableop_resource:	?M
9gru_56_while_gru_cell_56_matmul_1_readvariableop_resource:
????.gru_56/while/gru_cell_56/MatMul/ReadVariableOp?0gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp?'gru_56/while/gru_cell_56/ReadVariableOp?
>gru_56/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_56/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_56/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_56_while_tensorarrayv2read_tensorlistgetitem_gru_56_tensorarrayunstack_tensorlistfromtensor_0gru_56_while_placeholderGgru_56/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_56/while/TensorArrayV2Read/TensorListGetItem?
'gru_56/while/gru_cell_56/ReadVariableOpReadVariableOp2gru_56_while_gru_cell_56_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_56/while/gru_cell_56/ReadVariableOp?
 gru_56/while/gru_cell_56/unstackUnpack/gru_56/while/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_56/while/gru_cell_56/unstack?
.gru_56/while/gru_cell_56/MatMul/ReadVariableOpReadVariableOp9gru_56_while_gru_cell_56_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_56/while/gru_cell_56/MatMul/ReadVariableOp?
gru_56/while/gru_cell_56/MatMulMatMul7gru_56/while/TensorArrayV2Read/TensorListGetItem:item:06gru_56/while/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_56/while/gru_cell_56/MatMul?
 gru_56/while/gru_cell_56/BiasAddBiasAdd)gru_56/while/gru_cell_56/MatMul:product:0)gru_56/while/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_56/while/gru_cell_56/BiasAdd?
(gru_56/while/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_56/while/gru_cell_56/split/split_dim?
gru_56/while/gru_cell_56/splitSplit1gru_56/while/gru_cell_56/split/split_dim:output:0)gru_56/while/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_56/while/gru_cell_56/split?
0gru_56/while/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp;gru_56_while_gru_cell_56_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp?
!gru_56/while/gru_cell_56/MatMul_1MatMulgru_56_while_placeholder_28gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_56/while/gru_cell_56/MatMul_1?
"gru_56/while/gru_cell_56/BiasAdd_1BiasAdd+gru_56/while/gru_cell_56/MatMul_1:product:0)gru_56/while/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_56/while/gru_cell_56/BiasAdd_1?
gru_56/while/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2 
gru_56/while/gru_cell_56/Const?
*gru_56/while/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_56/while/gru_cell_56/split_1/split_dim?
 gru_56/while/gru_cell_56/split_1SplitV+gru_56/while/gru_cell_56/BiasAdd_1:output:0'gru_56/while/gru_cell_56/Const:output:03gru_56/while/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_56/while/gru_cell_56/split_1?
gru_56/while/gru_cell_56/addAddV2'gru_56/while/gru_cell_56/split:output:0)gru_56/while/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_56/while/gru_cell_56/add?
 gru_56/while/gru_cell_56/SigmoidSigmoid gru_56/while/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_56/while/gru_cell_56/Sigmoid?
gru_56/while/gru_cell_56/add_1AddV2'gru_56/while/gru_cell_56/split:output:1)gru_56/while/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_56/while/gru_cell_56/add_1?
"gru_56/while/gru_cell_56/Sigmoid_1Sigmoid"gru_56/while/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_56/while/gru_cell_56/Sigmoid_1?
gru_56/while/gru_cell_56/mulMul&gru_56/while/gru_cell_56/Sigmoid_1:y:0)gru_56/while/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_56/while/gru_cell_56/mul?
gru_56/while/gru_cell_56/add_2AddV2'gru_56/while/gru_cell_56/split:output:2 gru_56/while/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_56/while/gru_cell_56/add_2?
gru_56/while/gru_cell_56/ReluRelu"gru_56/while/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_56/while/gru_cell_56/Relu?
gru_56/while/gru_cell_56/mul_1Mul$gru_56/while/gru_cell_56/Sigmoid:y:0gru_56_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_56/while/gru_cell_56/mul_1?
gru_56/while/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_56/while/gru_cell_56/sub/x?
gru_56/while/gru_cell_56/subSub'gru_56/while/gru_cell_56/sub/x:output:0$gru_56/while/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_56/while/gru_cell_56/sub?
gru_56/while/gru_cell_56/mul_2Mul gru_56/while/gru_cell_56/sub:z:0+gru_56/while/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_56/while/gru_cell_56/mul_2?
gru_56/while/gru_cell_56/add_3AddV2"gru_56/while/gru_cell_56/mul_1:z:0"gru_56/while/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_56/while/gru_cell_56/add_3?
1gru_56/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_56_while_placeholder_1gru_56_while_placeholder"gru_56/while/gru_cell_56/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_56/while/TensorArrayV2Write/TensorListSetItemj
gru_56/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_56/while/add/y?
gru_56/while/addAddV2gru_56_while_placeholdergru_56/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_56/while/addn
gru_56/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_56/while/add_1/y?
gru_56/while/add_1AddV2&gru_56_while_gru_56_while_loop_countergru_56/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_56/while/add_1?
gru_56/while/IdentityIdentitygru_56/while/add_1:z:0^gru_56/while/NoOp*
T0*
_output_shapes
: 2
gru_56/while/Identity?
gru_56/while/Identity_1Identity,gru_56_while_gru_56_while_maximum_iterations^gru_56/while/NoOp*
T0*
_output_shapes
: 2
gru_56/while/Identity_1?
gru_56/while/Identity_2Identitygru_56/while/add:z:0^gru_56/while/NoOp*
T0*
_output_shapes
: 2
gru_56/while/Identity_2?
gru_56/while/Identity_3IdentityAgru_56/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_56/while/NoOp*
T0*
_output_shapes
: 2
gru_56/while/Identity_3?
gru_56/while/Identity_4Identity"gru_56/while/gru_cell_56/add_3:z:0^gru_56/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_56/while/Identity_4?
gru_56/while/NoOpNoOp/^gru_56/while/gru_cell_56/MatMul/ReadVariableOp1^gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp(^gru_56/while/gru_cell_56/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_56/while/NoOp"L
#gru_56_while_gru_56_strided_slice_1%gru_56_while_gru_56_strided_slice_1_0"x
9gru_56_while_gru_cell_56_matmul_1_readvariableop_resource;gru_56_while_gru_cell_56_matmul_1_readvariableop_resource_0"t
7gru_56_while_gru_cell_56_matmul_readvariableop_resource9gru_56_while_gru_cell_56_matmul_readvariableop_resource_0"f
0gru_56_while_gru_cell_56_readvariableop_resource2gru_56_while_gru_cell_56_readvariableop_resource_0"7
gru_56_while_identitygru_56/while/Identity:output:0";
gru_56_while_identity_1 gru_56/while/Identity_1:output:0";
gru_56_while_identity_2 gru_56/while/Identity_2:output:0";
gru_56_while_identity_3 gru_56/while/Identity_3:output:0";
gru_56_while_identity_4 gru_56/while/Identity_4:output:0"?
_gru_56_while_tensorarrayv2read_tensorlistgetitem_gru_56_tensorarrayunstack_tensorlistfromtensoragru_56_while_tensorarrayv2read_tensorlistgetitem_gru_56_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_56/while/gru_cell_56/MatMul/ReadVariableOp.gru_56/while/gru_cell_56/MatMul/ReadVariableOp2d
0gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp0gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp2R
'gru_56/while/gru_cell_56/ReadVariableOp'gru_56/while/gru_cell_56/ReadVariableOp: 
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
f
G__inference_dropout_94_layer_call_and_return_conditional_losses_3601400

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
?P
?	
gru_56_while_body_3599164*
&gru_56_while_gru_56_while_loop_counter0
,gru_56_while_gru_56_while_maximum_iterations
gru_56_while_placeholder
gru_56_while_placeholder_1
gru_56_while_placeholder_2)
%gru_56_while_gru_56_strided_slice_1_0e
agru_56_while_tensorarrayv2read_tensorlistgetitem_gru_56_tensorarrayunstack_tensorlistfromtensor_0E
2gru_56_while_gru_cell_56_readvariableop_resource_0:	?L
9gru_56_while_gru_cell_56_matmul_readvariableop_resource_0:	?O
;gru_56_while_gru_cell_56_matmul_1_readvariableop_resource_0:
??
gru_56_while_identity
gru_56_while_identity_1
gru_56_while_identity_2
gru_56_while_identity_3
gru_56_while_identity_4'
#gru_56_while_gru_56_strided_slice_1c
_gru_56_while_tensorarrayv2read_tensorlistgetitem_gru_56_tensorarrayunstack_tensorlistfromtensorC
0gru_56_while_gru_cell_56_readvariableop_resource:	?J
7gru_56_while_gru_cell_56_matmul_readvariableop_resource:	?M
9gru_56_while_gru_cell_56_matmul_1_readvariableop_resource:
????.gru_56/while/gru_cell_56/MatMul/ReadVariableOp?0gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp?'gru_56/while/gru_cell_56/ReadVariableOp?
>gru_56/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_56/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_56/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_56_while_tensorarrayv2read_tensorlistgetitem_gru_56_tensorarrayunstack_tensorlistfromtensor_0gru_56_while_placeholderGgru_56/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_56/while/TensorArrayV2Read/TensorListGetItem?
'gru_56/while/gru_cell_56/ReadVariableOpReadVariableOp2gru_56_while_gru_cell_56_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_56/while/gru_cell_56/ReadVariableOp?
 gru_56/while/gru_cell_56/unstackUnpack/gru_56/while/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_56/while/gru_cell_56/unstack?
.gru_56/while/gru_cell_56/MatMul/ReadVariableOpReadVariableOp9gru_56_while_gru_cell_56_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_56/while/gru_cell_56/MatMul/ReadVariableOp?
gru_56/while/gru_cell_56/MatMulMatMul7gru_56/while/TensorArrayV2Read/TensorListGetItem:item:06gru_56/while/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_56/while/gru_cell_56/MatMul?
 gru_56/while/gru_cell_56/BiasAddBiasAdd)gru_56/while/gru_cell_56/MatMul:product:0)gru_56/while/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_56/while/gru_cell_56/BiasAdd?
(gru_56/while/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_56/while/gru_cell_56/split/split_dim?
gru_56/while/gru_cell_56/splitSplit1gru_56/while/gru_cell_56/split/split_dim:output:0)gru_56/while/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_56/while/gru_cell_56/split?
0gru_56/while/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp;gru_56_while_gru_cell_56_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp?
!gru_56/while/gru_cell_56/MatMul_1MatMulgru_56_while_placeholder_28gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_56/while/gru_cell_56/MatMul_1?
"gru_56/while/gru_cell_56/BiasAdd_1BiasAdd+gru_56/while/gru_cell_56/MatMul_1:product:0)gru_56/while/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_56/while/gru_cell_56/BiasAdd_1?
gru_56/while/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2 
gru_56/while/gru_cell_56/Const?
*gru_56/while/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_56/while/gru_cell_56/split_1/split_dim?
 gru_56/while/gru_cell_56/split_1SplitV+gru_56/while/gru_cell_56/BiasAdd_1:output:0'gru_56/while/gru_cell_56/Const:output:03gru_56/while/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_56/while/gru_cell_56/split_1?
gru_56/while/gru_cell_56/addAddV2'gru_56/while/gru_cell_56/split:output:0)gru_56/while/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_56/while/gru_cell_56/add?
 gru_56/while/gru_cell_56/SigmoidSigmoid gru_56/while/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_56/while/gru_cell_56/Sigmoid?
gru_56/while/gru_cell_56/add_1AddV2'gru_56/while/gru_cell_56/split:output:1)gru_56/while/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_56/while/gru_cell_56/add_1?
"gru_56/while/gru_cell_56/Sigmoid_1Sigmoid"gru_56/while/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_56/while/gru_cell_56/Sigmoid_1?
gru_56/while/gru_cell_56/mulMul&gru_56/while/gru_cell_56/Sigmoid_1:y:0)gru_56/while/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_56/while/gru_cell_56/mul?
gru_56/while/gru_cell_56/add_2AddV2'gru_56/while/gru_cell_56/split:output:2 gru_56/while/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_56/while/gru_cell_56/add_2?
gru_56/while/gru_cell_56/ReluRelu"gru_56/while/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_56/while/gru_cell_56/Relu?
gru_56/while/gru_cell_56/mul_1Mul$gru_56/while/gru_cell_56/Sigmoid:y:0gru_56_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_56/while/gru_cell_56/mul_1?
gru_56/while/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_56/while/gru_cell_56/sub/x?
gru_56/while/gru_cell_56/subSub'gru_56/while/gru_cell_56/sub/x:output:0$gru_56/while/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_56/while/gru_cell_56/sub?
gru_56/while/gru_cell_56/mul_2Mul gru_56/while/gru_cell_56/sub:z:0+gru_56/while/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_56/while/gru_cell_56/mul_2?
gru_56/while/gru_cell_56/add_3AddV2"gru_56/while/gru_cell_56/mul_1:z:0"gru_56/while/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_56/while/gru_cell_56/add_3?
1gru_56/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_56_while_placeholder_1gru_56_while_placeholder"gru_56/while/gru_cell_56/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_56/while/TensorArrayV2Write/TensorListSetItemj
gru_56/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_56/while/add/y?
gru_56/while/addAddV2gru_56_while_placeholdergru_56/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_56/while/addn
gru_56/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_56/while/add_1/y?
gru_56/while/add_1AddV2&gru_56_while_gru_56_while_loop_countergru_56/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_56/while/add_1?
gru_56/while/IdentityIdentitygru_56/while/add_1:z:0^gru_56/while/NoOp*
T0*
_output_shapes
: 2
gru_56/while/Identity?
gru_56/while/Identity_1Identity,gru_56_while_gru_56_while_maximum_iterations^gru_56/while/NoOp*
T0*
_output_shapes
: 2
gru_56/while/Identity_1?
gru_56/while/Identity_2Identitygru_56/while/add:z:0^gru_56/while/NoOp*
T0*
_output_shapes
: 2
gru_56/while/Identity_2?
gru_56/while/Identity_3IdentityAgru_56/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_56/while/NoOp*
T0*
_output_shapes
: 2
gru_56/while/Identity_3?
gru_56/while/Identity_4Identity"gru_56/while/gru_cell_56/add_3:z:0^gru_56/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_56/while/Identity_4?
gru_56/while/NoOpNoOp/^gru_56/while/gru_cell_56/MatMul/ReadVariableOp1^gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp(^gru_56/while/gru_cell_56/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_56/while/NoOp"L
#gru_56_while_gru_56_strided_slice_1%gru_56_while_gru_56_strided_slice_1_0"x
9gru_56_while_gru_cell_56_matmul_1_readvariableop_resource;gru_56_while_gru_cell_56_matmul_1_readvariableop_resource_0"t
7gru_56_while_gru_cell_56_matmul_readvariableop_resource9gru_56_while_gru_cell_56_matmul_readvariableop_resource_0"f
0gru_56_while_gru_cell_56_readvariableop_resource2gru_56_while_gru_cell_56_readvariableop_resource_0"7
gru_56_while_identitygru_56/while/Identity:output:0";
gru_56_while_identity_1 gru_56/while/Identity_1:output:0";
gru_56_while_identity_2 gru_56/while/Identity_2:output:0";
gru_56_while_identity_3 gru_56/while/Identity_3:output:0";
gru_56_while_identity_4 gru_56/while/Identity_4:output:0"?
_gru_56_while_tensorarrayv2read_tensorlistgetitem_gru_56_tensorarrayunstack_tensorlistfromtensoragru_56_while_tensorarrayv2read_tensorlistgetitem_gru_56_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_56/while/gru_cell_56/MatMul/ReadVariableOp.gru_56/while/gru_cell_56/MatMul/ReadVariableOp2d
0gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp0gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp2R
'gru_56/while/gru_cell_56/ReadVariableOp'gru_56/while/gru_cell_56/ReadVariableOp: 
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
?Y
?
C__inference_gru_56_layer_call_and_return_conditional_losses_3600250
inputs_06
#gru_cell_56_readvariableop_resource:	?=
*gru_cell_56_matmul_readvariableop_resource:	?@
,gru_cell_56_matmul_1_readvariableop_resource:
??
identity??!gru_cell_56/MatMul/ReadVariableOp?#gru_cell_56/MatMul_1/ReadVariableOp?gru_cell_56/ReadVariableOp?whileF
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
gru_cell_56/ReadVariableOpReadVariableOp#gru_cell_56_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_56/ReadVariableOp?
gru_cell_56/unstackUnpack"gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_56/unstack?
!gru_cell_56/MatMul/ReadVariableOpReadVariableOp*gru_cell_56_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_56/MatMul/ReadVariableOp?
gru_cell_56/MatMulMatMulstrided_slice_2:output:0)gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul?
gru_cell_56/BiasAddBiasAddgru_cell_56/MatMul:product:0gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd?
gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split/split_dim?
gru_cell_56/splitSplit$gru_cell_56/split/split_dim:output:0gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split?
#gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_56_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_56/MatMul_1/ReadVariableOp?
gru_cell_56/MatMul_1MatMulzeros:output:0+gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul_1?
gru_cell_56/BiasAdd_1BiasAddgru_cell_56/MatMul_1:product:0gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd_1{
gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_56/Const?
gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split_1/split_dim?
gru_cell_56/split_1SplitVgru_cell_56/BiasAdd_1:output:0gru_cell_56/Const:output:0&gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split_1?
gru_cell_56/addAddV2gru_cell_56/split:output:0gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add}
gru_cell_56/SigmoidSigmoidgru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid?
gru_cell_56/add_1AddV2gru_cell_56/split:output:1gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_1?
gru_cell_56/Sigmoid_1Sigmoidgru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid_1?
gru_cell_56/mulMulgru_cell_56/Sigmoid_1:y:0gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul?
gru_cell_56/add_2AddV2gru_cell_56/split:output:2gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_2v
gru_cell_56/ReluRelugru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Relu?
gru_cell_56/mul_1Mulgru_cell_56/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_1k
gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_56/sub/x?
gru_cell_56/subSubgru_cell_56/sub/x:output:0gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/sub?
gru_cell_56/mul_2Mulgru_cell_56/sub:z:0gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_2?
gru_cell_56/add_3AddV2gru_cell_56/mul_1:z:0gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_56_readvariableop_resource*gru_cell_56_matmul_readvariableop_resource,gru_cell_56_matmul_1_readvariableop_resource*
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
bodyR
while_body_3600161*
condR
while_cond_3600160*9
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
NoOpNoOp"^gru_cell_56/MatMul/ReadVariableOp$^gru_cell_56/MatMul_1/ReadVariableOp^gru_cell_56/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2F
!gru_cell_56/MatMul/ReadVariableOp!gru_cell_56/MatMul/ReadVariableOp2J
#gru_cell_56/MatMul_1/ReadVariableOp#gru_cell_56/MatMul_1/ReadVariableOp28
gru_cell_56/ReadVariableOpgru_cell_56/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
e
G__inference_dropout_91_layer_call_and_return_conditional_losses_3600571

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
f
G__inference_dropout_91_layer_call_and_return_conditional_losses_3600583

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
?
while_cond_3600843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3600843___redundant_placeholder05
1while_while_cond_3600843___redundant_placeholder15
1while_while_cond_3600843___redundant_placeholder25
1while_while_cond_3600843___redundant_placeholder3
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
e
G__inference_dropout_94_layer_call_and_return_conditional_losses_3598269

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
?;
?
C__inference_gru_56_layer_call_and_return_conditional_losses_3596857

inputs&
gru_cell_56_3596781:	?&
gru_cell_56_3596783:	?'
gru_cell_56_3596785:
??
identity??#gru_cell_56/StatefulPartitionedCall?whileD
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
#gru_cell_56/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_56_3596781gru_cell_56_3596783gru_cell_56_3596785*
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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_35967802%
#gru_cell_56/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_56_3596781gru_cell_56_3596783gru_cell_56_3596785*
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
bodyR
while_body_3596793*
condR
while_cond_3596792*9
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
NoOpNoOp$^gru_cell_56/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#gru_cell_56/StatefulPartitionedCall#gru_cell_56/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_92_layer_call_and_return_conditional_losses_3598431

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
?
f
G__inference_dropout_93_layer_call_and_return_conditional_losses_3598398

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
e
,__inference_dropout_92_layer_call_fn_3601249

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
GPU 2J 8? *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_35984312
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
? 
?
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_3601651

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
? 
?
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_3601612

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
??
?
"__inference__wrapped_model_3596710
gru_56_inputK
8sequential_28_gru_56_gru_cell_56_readvariableop_resource:	?R
?sequential_28_gru_56_gru_cell_56_matmul_readvariableop_resource:	?U
Asequential_28_gru_56_gru_cell_56_matmul_1_readvariableop_resource:
??K
8sequential_28_gru_57_gru_cell_57_readvariableop_resource:	?S
?sequential_28_gru_57_gru_cell_57_matmul_readvariableop_resource:
??U
Asequential_28_gru_57_gru_cell_57_matmul_1_readvariableop_resource:
??L
8sequential_28_dense_63_tensordot_readvariableop_resource:
??E
6sequential_28_dense_63_biasadd_readvariableop_resource:	?L
8sequential_28_dense_64_tensordot_readvariableop_resource:
??E
6sequential_28_dense_64_biasadd_readvariableop_resource:	?K
8sequential_28_dense_65_tensordot_readvariableop_resource:	?D
6sequential_28_dense_65_biasadd_readvariableop_resource:
identity??-sequential_28/dense_63/BiasAdd/ReadVariableOp?/sequential_28/dense_63/Tensordot/ReadVariableOp?-sequential_28/dense_64/BiasAdd/ReadVariableOp?/sequential_28/dense_64/Tensordot/ReadVariableOp?-sequential_28/dense_65/BiasAdd/ReadVariableOp?/sequential_28/dense_65/Tensordot/ReadVariableOp?6sequential_28/gru_56/gru_cell_56/MatMul/ReadVariableOp?8sequential_28/gru_56/gru_cell_56/MatMul_1/ReadVariableOp?/sequential_28/gru_56/gru_cell_56/ReadVariableOp?sequential_28/gru_56/while?6sequential_28/gru_57/gru_cell_57/MatMul/ReadVariableOp?8sequential_28/gru_57/gru_cell_57/MatMul_1/ReadVariableOp?/sequential_28/gru_57/gru_cell_57/ReadVariableOp?sequential_28/gru_57/whilet
sequential_28/gru_56/ShapeShapegru_56_input*
T0*
_output_shapes
:2
sequential_28/gru_56/Shape?
(sequential_28/gru_56/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_28/gru_56/strided_slice/stack?
*sequential_28/gru_56/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_28/gru_56/strided_slice/stack_1?
*sequential_28/gru_56/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_28/gru_56/strided_slice/stack_2?
"sequential_28/gru_56/strided_sliceStridedSlice#sequential_28/gru_56/Shape:output:01sequential_28/gru_56/strided_slice/stack:output:03sequential_28/gru_56/strided_slice/stack_1:output:03sequential_28/gru_56/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_28/gru_56/strided_slice?
#sequential_28/gru_56/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_28/gru_56/zeros/packed/1?
!sequential_28/gru_56/zeros/packedPack+sequential_28/gru_56/strided_slice:output:0,sequential_28/gru_56/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_28/gru_56/zeros/packed?
 sequential_28/gru_56/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_28/gru_56/zeros/Const?
sequential_28/gru_56/zerosFill*sequential_28/gru_56/zeros/packed:output:0)sequential_28/gru_56/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_28/gru_56/zeros?
#sequential_28/gru_56/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_28/gru_56/transpose/perm?
sequential_28/gru_56/transpose	Transposegru_56_input,sequential_28/gru_56/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2 
sequential_28/gru_56/transpose?
sequential_28/gru_56/Shape_1Shape"sequential_28/gru_56/transpose:y:0*
T0*
_output_shapes
:2
sequential_28/gru_56/Shape_1?
*sequential_28/gru_56/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_28/gru_56/strided_slice_1/stack?
,sequential_28/gru_56/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_56/strided_slice_1/stack_1?
,sequential_28/gru_56/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_56/strided_slice_1/stack_2?
$sequential_28/gru_56/strided_slice_1StridedSlice%sequential_28/gru_56/Shape_1:output:03sequential_28/gru_56/strided_slice_1/stack:output:05sequential_28/gru_56/strided_slice_1/stack_1:output:05sequential_28/gru_56/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_28/gru_56/strided_slice_1?
0sequential_28/gru_56/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_28/gru_56/TensorArrayV2/element_shape?
"sequential_28/gru_56/TensorArrayV2TensorListReserve9sequential_28/gru_56/TensorArrayV2/element_shape:output:0-sequential_28/gru_56/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_28/gru_56/TensorArrayV2?
Jsequential_28/gru_56/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jsequential_28/gru_56/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_28/gru_56/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_28/gru_56/transpose:y:0Ssequential_28/gru_56/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_28/gru_56/TensorArrayUnstack/TensorListFromTensor?
*sequential_28/gru_56/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_28/gru_56/strided_slice_2/stack?
,sequential_28/gru_56/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_56/strided_slice_2/stack_1?
,sequential_28/gru_56/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_56/strided_slice_2/stack_2?
$sequential_28/gru_56/strided_slice_2StridedSlice"sequential_28/gru_56/transpose:y:03sequential_28/gru_56/strided_slice_2/stack:output:05sequential_28/gru_56/strided_slice_2/stack_1:output:05sequential_28/gru_56/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2&
$sequential_28/gru_56/strided_slice_2?
/sequential_28/gru_56/gru_cell_56/ReadVariableOpReadVariableOp8sequential_28_gru_56_gru_cell_56_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_28/gru_56/gru_cell_56/ReadVariableOp?
(sequential_28/gru_56/gru_cell_56/unstackUnpack7sequential_28/gru_56/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2*
(sequential_28/gru_56/gru_cell_56/unstack?
6sequential_28/gru_56/gru_cell_56/MatMul/ReadVariableOpReadVariableOp?sequential_28_gru_56_gru_cell_56_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype028
6sequential_28/gru_56/gru_cell_56/MatMul/ReadVariableOp?
'sequential_28/gru_56/gru_cell_56/MatMulMatMul-sequential_28/gru_56/strided_slice_2:output:0>sequential_28/gru_56/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_28/gru_56/gru_cell_56/MatMul?
(sequential_28/gru_56/gru_cell_56/BiasAddBiasAdd1sequential_28/gru_56/gru_cell_56/MatMul:product:01sequential_28/gru_56/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2*
(sequential_28/gru_56/gru_cell_56/BiasAdd?
0sequential_28/gru_56/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_28/gru_56/gru_cell_56/split/split_dim?
&sequential_28/gru_56/gru_cell_56/splitSplit9sequential_28/gru_56/gru_cell_56/split/split_dim:output:01sequential_28/gru_56/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2(
&sequential_28/gru_56/gru_cell_56/split?
8sequential_28/gru_56/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOpAsequential_28_gru_56_gru_cell_56_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8sequential_28/gru_56/gru_cell_56/MatMul_1/ReadVariableOp?
)sequential_28/gru_56/gru_cell_56/MatMul_1MatMul#sequential_28/gru_56/zeros:output:0@sequential_28/gru_56/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_28/gru_56/gru_cell_56/MatMul_1?
*sequential_28/gru_56/gru_cell_56/BiasAdd_1BiasAdd3sequential_28/gru_56/gru_cell_56/MatMul_1:product:01sequential_28/gru_56/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2,
*sequential_28/gru_56/gru_cell_56/BiasAdd_1?
&sequential_28/gru_56/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2(
&sequential_28/gru_56/gru_cell_56/Const?
2sequential_28/gru_56/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2sequential_28/gru_56/gru_cell_56/split_1/split_dim?
(sequential_28/gru_56/gru_cell_56/split_1SplitV3sequential_28/gru_56/gru_cell_56/BiasAdd_1:output:0/sequential_28/gru_56/gru_cell_56/Const:output:0;sequential_28/gru_56/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2*
(sequential_28/gru_56/gru_cell_56/split_1?
$sequential_28/gru_56/gru_cell_56/addAddV2/sequential_28/gru_56/gru_cell_56/split:output:01sequential_28/gru_56/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2&
$sequential_28/gru_56/gru_cell_56/add?
(sequential_28/gru_56/gru_cell_56/SigmoidSigmoid(sequential_28/gru_56/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_28/gru_56/gru_cell_56/Sigmoid?
&sequential_28/gru_56/gru_cell_56/add_1AddV2/sequential_28/gru_56/gru_cell_56/split:output:11sequential_28/gru_56/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2(
&sequential_28/gru_56/gru_cell_56/add_1?
*sequential_28/gru_56/gru_cell_56/Sigmoid_1Sigmoid*sequential_28/gru_56/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2,
*sequential_28/gru_56/gru_cell_56/Sigmoid_1?
$sequential_28/gru_56/gru_cell_56/mulMul.sequential_28/gru_56/gru_cell_56/Sigmoid_1:y:01sequential_28/gru_56/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2&
$sequential_28/gru_56/gru_cell_56/mul?
&sequential_28/gru_56/gru_cell_56/add_2AddV2/sequential_28/gru_56/gru_cell_56/split:output:2(sequential_28/gru_56/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2(
&sequential_28/gru_56/gru_cell_56/add_2?
%sequential_28/gru_56/gru_cell_56/ReluRelu*sequential_28/gru_56/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2'
%sequential_28/gru_56/gru_cell_56/Relu?
&sequential_28/gru_56/gru_cell_56/mul_1Mul,sequential_28/gru_56/gru_cell_56/Sigmoid:y:0#sequential_28/gru_56/zeros:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_28/gru_56/gru_cell_56/mul_1?
&sequential_28/gru_56/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&sequential_28/gru_56/gru_cell_56/sub/x?
$sequential_28/gru_56/gru_cell_56/subSub/sequential_28/gru_56/gru_cell_56/sub/x:output:0,sequential_28/gru_56/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2&
$sequential_28/gru_56/gru_cell_56/sub?
&sequential_28/gru_56/gru_cell_56/mul_2Mul(sequential_28/gru_56/gru_cell_56/sub:z:03sequential_28/gru_56/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2(
&sequential_28/gru_56/gru_cell_56/mul_2?
&sequential_28/gru_56/gru_cell_56/add_3AddV2*sequential_28/gru_56/gru_cell_56/mul_1:z:0*sequential_28/gru_56/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2(
&sequential_28/gru_56/gru_cell_56/add_3?
2sequential_28/gru_56/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   24
2sequential_28/gru_56/TensorArrayV2_1/element_shape?
$sequential_28/gru_56/TensorArrayV2_1TensorListReserve;sequential_28/gru_56/TensorArrayV2_1/element_shape:output:0-sequential_28/gru_56/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_28/gru_56/TensorArrayV2_1x
sequential_28/gru_56/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_28/gru_56/time?
-sequential_28/gru_56/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_28/gru_56/while/maximum_iterations?
'sequential_28/gru_56/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_28/gru_56/while/loop_counter?
sequential_28/gru_56/whileWhile0sequential_28/gru_56/while/loop_counter:output:06sequential_28/gru_56/while/maximum_iterations:output:0"sequential_28/gru_56/time:output:0-sequential_28/gru_56/TensorArrayV2_1:handle:0#sequential_28/gru_56/zeros:output:0-sequential_28/gru_56/strided_slice_1:output:0Lsequential_28/gru_56/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_28_gru_56_gru_cell_56_readvariableop_resource?sequential_28_gru_56_gru_cell_56_matmul_readvariableop_resourceAsequential_28_gru_56_gru_cell_56_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *3
body+R)
'sequential_28_gru_56_while_body_3596388*3
cond+R)
'sequential_28_gru_56_while_cond_3596387*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
sequential_28/gru_56/while?
Esequential_28/gru_56/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
Esequential_28/gru_56/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_28/gru_56/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_28/gru_56/while:output:3Nsequential_28/gru_56/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype029
7sequential_28/gru_56/TensorArrayV2Stack/TensorListStack?
*sequential_28/gru_56/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_28/gru_56/strided_slice_3/stack?
,sequential_28/gru_56/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_28/gru_56/strided_slice_3/stack_1?
,sequential_28/gru_56/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_56/strided_slice_3/stack_2?
$sequential_28/gru_56/strided_slice_3StridedSlice@sequential_28/gru_56/TensorArrayV2Stack/TensorListStack:tensor:03sequential_28/gru_56/strided_slice_3/stack:output:05sequential_28/gru_56/strided_slice_3/stack_1:output:05sequential_28/gru_56/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_28/gru_56/strided_slice_3?
%sequential_28/gru_56/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_28/gru_56/transpose_1/perm?
 sequential_28/gru_56/transpose_1	Transpose@sequential_28/gru_56/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_28/gru_56/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_28/gru_56/transpose_1?
sequential_28/gru_56/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_28/gru_56/runtime?
!sequential_28/dropout_91/IdentityIdentity$sequential_28/gru_56/transpose_1:y:0*
T0*,
_output_shapes
:??????????2#
!sequential_28/dropout_91/Identity?
sequential_28/gru_57/ShapeShape*sequential_28/dropout_91/Identity:output:0*
T0*
_output_shapes
:2
sequential_28/gru_57/Shape?
(sequential_28/gru_57/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_28/gru_57/strided_slice/stack?
*sequential_28/gru_57/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_28/gru_57/strided_slice/stack_1?
*sequential_28/gru_57/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_28/gru_57/strided_slice/stack_2?
"sequential_28/gru_57/strided_sliceStridedSlice#sequential_28/gru_57/Shape:output:01sequential_28/gru_57/strided_slice/stack:output:03sequential_28/gru_57/strided_slice/stack_1:output:03sequential_28/gru_57/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_28/gru_57/strided_slice?
#sequential_28/gru_57/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_28/gru_57/zeros/packed/1?
!sequential_28/gru_57/zeros/packedPack+sequential_28/gru_57/strided_slice:output:0,sequential_28/gru_57/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_28/gru_57/zeros/packed?
 sequential_28/gru_57/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_28/gru_57/zeros/Const?
sequential_28/gru_57/zerosFill*sequential_28/gru_57/zeros/packed:output:0)sequential_28/gru_57/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_28/gru_57/zeros?
#sequential_28/gru_57/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_28/gru_57/transpose/perm?
sequential_28/gru_57/transpose	Transpose*sequential_28/dropout_91/Identity:output:0,sequential_28/gru_57/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2 
sequential_28/gru_57/transpose?
sequential_28/gru_57/Shape_1Shape"sequential_28/gru_57/transpose:y:0*
T0*
_output_shapes
:2
sequential_28/gru_57/Shape_1?
*sequential_28/gru_57/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_28/gru_57/strided_slice_1/stack?
,sequential_28/gru_57/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_57/strided_slice_1/stack_1?
,sequential_28/gru_57/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_57/strided_slice_1/stack_2?
$sequential_28/gru_57/strided_slice_1StridedSlice%sequential_28/gru_57/Shape_1:output:03sequential_28/gru_57/strided_slice_1/stack:output:05sequential_28/gru_57/strided_slice_1/stack_1:output:05sequential_28/gru_57/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_28/gru_57/strided_slice_1?
0sequential_28/gru_57/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_28/gru_57/TensorArrayV2/element_shape?
"sequential_28/gru_57/TensorArrayV2TensorListReserve9sequential_28/gru_57/TensorArrayV2/element_shape:output:0-sequential_28/gru_57/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_28/gru_57/TensorArrayV2?
Jsequential_28/gru_57/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jsequential_28/gru_57/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_28/gru_57/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_28/gru_57/transpose:y:0Ssequential_28/gru_57/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_28/gru_57/TensorArrayUnstack/TensorListFromTensor?
*sequential_28/gru_57/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_28/gru_57/strided_slice_2/stack?
,sequential_28/gru_57/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_57/strided_slice_2/stack_1?
,sequential_28/gru_57/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_57/strided_slice_2/stack_2?
$sequential_28/gru_57/strided_slice_2StridedSlice"sequential_28/gru_57/transpose:y:03sequential_28/gru_57/strided_slice_2/stack:output:05sequential_28/gru_57/strided_slice_2/stack_1:output:05sequential_28/gru_57/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_28/gru_57/strided_slice_2?
/sequential_28/gru_57/gru_cell_57/ReadVariableOpReadVariableOp8sequential_28_gru_57_gru_cell_57_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_28/gru_57/gru_cell_57/ReadVariableOp?
(sequential_28/gru_57/gru_cell_57/unstackUnpack7sequential_28/gru_57/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2*
(sequential_28/gru_57/gru_cell_57/unstack?
6sequential_28/gru_57/gru_cell_57/MatMul/ReadVariableOpReadVariableOp?sequential_28_gru_57_gru_cell_57_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6sequential_28/gru_57/gru_cell_57/MatMul/ReadVariableOp?
'sequential_28/gru_57/gru_cell_57/MatMulMatMul-sequential_28/gru_57/strided_slice_2:output:0>sequential_28/gru_57/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_28/gru_57/gru_cell_57/MatMul?
(sequential_28/gru_57/gru_cell_57/BiasAddBiasAdd1sequential_28/gru_57/gru_cell_57/MatMul:product:01sequential_28/gru_57/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2*
(sequential_28/gru_57/gru_cell_57/BiasAdd?
0sequential_28/gru_57/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_28/gru_57/gru_cell_57/split/split_dim?
&sequential_28/gru_57/gru_cell_57/splitSplit9sequential_28/gru_57/gru_cell_57/split/split_dim:output:01sequential_28/gru_57/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2(
&sequential_28/gru_57/gru_cell_57/split?
8sequential_28/gru_57/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOpAsequential_28_gru_57_gru_cell_57_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8sequential_28/gru_57/gru_cell_57/MatMul_1/ReadVariableOp?
)sequential_28/gru_57/gru_cell_57/MatMul_1MatMul#sequential_28/gru_57/zeros:output:0@sequential_28/gru_57/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_28/gru_57/gru_cell_57/MatMul_1?
*sequential_28/gru_57/gru_cell_57/BiasAdd_1BiasAdd3sequential_28/gru_57/gru_cell_57/MatMul_1:product:01sequential_28/gru_57/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2,
*sequential_28/gru_57/gru_cell_57/BiasAdd_1?
&sequential_28/gru_57/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2(
&sequential_28/gru_57/gru_cell_57/Const?
2sequential_28/gru_57/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2sequential_28/gru_57/gru_cell_57/split_1/split_dim?
(sequential_28/gru_57/gru_cell_57/split_1SplitV3sequential_28/gru_57/gru_cell_57/BiasAdd_1:output:0/sequential_28/gru_57/gru_cell_57/Const:output:0;sequential_28/gru_57/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2*
(sequential_28/gru_57/gru_cell_57/split_1?
$sequential_28/gru_57/gru_cell_57/addAddV2/sequential_28/gru_57/gru_cell_57/split:output:01sequential_28/gru_57/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2&
$sequential_28/gru_57/gru_cell_57/add?
(sequential_28/gru_57/gru_cell_57/SigmoidSigmoid(sequential_28/gru_57/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_28/gru_57/gru_cell_57/Sigmoid?
&sequential_28/gru_57/gru_cell_57/add_1AddV2/sequential_28/gru_57/gru_cell_57/split:output:11sequential_28/gru_57/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2(
&sequential_28/gru_57/gru_cell_57/add_1?
*sequential_28/gru_57/gru_cell_57/Sigmoid_1Sigmoid*sequential_28/gru_57/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2,
*sequential_28/gru_57/gru_cell_57/Sigmoid_1?
$sequential_28/gru_57/gru_cell_57/mulMul.sequential_28/gru_57/gru_cell_57/Sigmoid_1:y:01sequential_28/gru_57/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2&
$sequential_28/gru_57/gru_cell_57/mul?
&sequential_28/gru_57/gru_cell_57/add_2AddV2/sequential_28/gru_57/gru_cell_57/split:output:2(sequential_28/gru_57/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2(
&sequential_28/gru_57/gru_cell_57/add_2?
%sequential_28/gru_57/gru_cell_57/ReluRelu*sequential_28/gru_57/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2'
%sequential_28/gru_57/gru_cell_57/Relu?
&sequential_28/gru_57/gru_cell_57/mul_1Mul,sequential_28/gru_57/gru_cell_57/Sigmoid:y:0#sequential_28/gru_57/zeros:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_28/gru_57/gru_cell_57/mul_1?
&sequential_28/gru_57/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&sequential_28/gru_57/gru_cell_57/sub/x?
$sequential_28/gru_57/gru_cell_57/subSub/sequential_28/gru_57/gru_cell_57/sub/x:output:0,sequential_28/gru_57/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2&
$sequential_28/gru_57/gru_cell_57/sub?
&sequential_28/gru_57/gru_cell_57/mul_2Mul(sequential_28/gru_57/gru_cell_57/sub:z:03sequential_28/gru_57/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2(
&sequential_28/gru_57/gru_cell_57/mul_2?
&sequential_28/gru_57/gru_cell_57/add_3AddV2*sequential_28/gru_57/gru_cell_57/mul_1:z:0*sequential_28/gru_57/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2(
&sequential_28/gru_57/gru_cell_57/add_3?
2sequential_28/gru_57/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   24
2sequential_28/gru_57/TensorArrayV2_1/element_shape?
$sequential_28/gru_57/TensorArrayV2_1TensorListReserve;sequential_28/gru_57/TensorArrayV2_1/element_shape:output:0-sequential_28/gru_57/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_28/gru_57/TensorArrayV2_1x
sequential_28/gru_57/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_28/gru_57/time?
-sequential_28/gru_57/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_28/gru_57/while/maximum_iterations?
'sequential_28/gru_57/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_28/gru_57/while/loop_counter?
sequential_28/gru_57/whileWhile0sequential_28/gru_57/while/loop_counter:output:06sequential_28/gru_57/while/maximum_iterations:output:0"sequential_28/gru_57/time:output:0-sequential_28/gru_57/TensorArrayV2_1:handle:0#sequential_28/gru_57/zeros:output:0-sequential_28/gru_57/strided_slice_1:output:0Lsequential_28/gru_57/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_28_gru_57_gru_cell_57_readvariableop_resource?sequential_28_gru_57_gru_cell_57_matmul_readvariableop_resourceAsequential_28_gru_57_gru_cell_57_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *3
body+R)
'sequential_28_gru_57_while_body_3596538*3
cond+R)
'sequential_28_gru_57_while_cond_3596537*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
sequential_28/gru_57/while?
Esequential_28/gru_57/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
Esequential_28/gru_57/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_28/gru_57/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_28/gru_57/while:output:3Nsequential_28/gru_57/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype029
7sequential_28/gru_57/TensorArrayV2Stack/TensorListStack?
*sequential_28/gru_57/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_28/gru_57/strided_slice_3/stack?
,sequential_28/gru_57/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_28/gru_57/strided_slice_3/stack_1?
,sequential_28/gru_57/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_57/strided_slice_3/stack_2?
$sequential_28/gru_57/strided_slice_3StridedSlice@sequential_28/gru_57/TensorArrayV2Stack/TensorListStack:tensor:03sequential_28/gru_57/strided_slice_3/stack:output:05sequential_28/gru_57/strided_slice_3/stack_1:output:05sequential_28/gru_57/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_28/gru_57/strided_slice_3?
%sequential_28/gru_57/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_28/gru_57/transpose_1/perm?
 sequential_28/gru_57/transpose_1	Transpose@sequential_28/gru_57/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_28/gru_57/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_28/gru_57/transpose_1?
sequential_28/gru_57/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_28/gru_57/runtime?
!sequential_28/dropout_92/IdentityIdentity$sequential_28/gru_57/transpose_1:y:0*
T0*,
_output_shapes
:??????????2#
!sequential_28/dropout_92/Identity?
/sequential_28/dense_63/Tensordot/ReadVariableOpReadVariableOp8sequential_28_dense_63_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential_28/dense_63/Tensordot/ReadVariableOp?
%sequential_28/dense_63/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_28/dense_63/Tensordot/axes?
%sequential_28/dense_63/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_28/dense_63/Tensordot/free?
&sequential_28/dense_63/Tensordot/ShapeShape*sequential_28/dropout_92/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_28/dense_63/Tensordot/Shape?
.sequential_28/dense_63/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_28/dense_63/Tensordot/GatherV2/axis?
)sequential_28/dense_63/Tensordot/GatherV2GatherV2/sequential_28/dense_63/Tensordot/Shape:output:0.sequential_28/dense_63/Tensordot/free:output:07sequential_28/dense_63/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_28/dense_63/Tensordot/GatherV2?
0sequential_28/dense_63/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_28/dense_63/Tensordot/GatherV2_1/axis?
+sequential_28/dense_63/Tensordot/GatherV2_1GatherV2/sequential_28/dense_63/Tensordot/Shape:output:0.sequential_28/dense_63/Tensordot/axes:output:09sequential_28/dense_63/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_28/dense_63/Tensordot/GatherV2_1?
&sequential_28/dense_63/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_28/dense_63/Tensordot/Const?
%sequential_28/dense_63/Tensordot/ProdProd2sequential_28/dense_63/Tensordot/GatherV2:output:0/sequential_28/dense_63/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_28/dense_63/Tensordot/Prod?
(sequential_28/dense_63/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_28/dense_63/Tensordot/Const_1?
'sequential_28/dense_63/Tensordot/Prod_1Prod4sequential_28/dense_63/Tensordot/GatherV2_1:output:01sequential_28/dense_63/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_28/dense_63/Tensordot/Prod_1?
,sequential_28/dense_63/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_28/dense_63/Tensordot/concat/axis?
'sequential_28/dense_63/Tensordot/concatConcatV2.sequential_28/dense_63/Tensordot/free:output:0.sequential_28/dense_63/Tensordot/axes:output:05sequential_28/dense_63/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_28/dense_63/Tensordot/concat?
&sequential_28/dense_63/Tensordot/stackPack.sequential_28/dense_63/Tensordot/Prod:output:00sequential_28/dense_63/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_28/dense_63/Tensordot/stack?
*sequential_28/dense_63/Tensordot/transpose	Transpose*sequential_28/dropout_92/Identity:output:00sequential_28/dense_63/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2,
*sequential_28/dense_63/Tensordot/transpose?
(sequential_28/dense_63/Tensordot/ReshapeReshape.sequential_28/dense_63/Tensordot/transpose:y:0/sequential_28/dense_63/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_28/dense_63/Tensordot/Reshape?
'sequential_28/dense_63/Tensordot/MatMulMatMul1sequential_28/dense_63/Tensordot/Reshape:output:07sequential_28/dense_63/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_28/dense_63/Tensordot/MatMul?
(sequential_28/dense_63/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2*
(sequential_28/dense_63/Tensordot/Const_2?
.sequential_28/dense_63/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_28/dense_63/Tensordot/concat_1/axis?
)sequential_28/dense_63/Tensordot/concat_1ConcatV22sequential_28/dense_63/Tensordot/GatherV2:output:01sequential_28/dense_63/Tensordot/Const_2:output:07sequential_28/dense_63/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_28/dense_63/Tensordot/concat_1?
 sequential_28/dense_63/TensordotReshape1sequential_28/dense_63/Tensordot/MatMul:product:02sequential_28/dense_63/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_28/dense_63/Tensordot?
-sequential_28/dense_63/BiasAdd/ReadVariableOpReadVariableOp6sequential_28_dense_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_28/dense_63/BiasAdd/ReadVariableOp?
sequential_28/dense_63/BiasAddBiasAdd)sequential_28/dense_63/Tensordot:output:05sequential_28/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2 
sequential_28/dense_63/BiasAdd?
sequential_28/dense_63/ReluRelu'sequential_28/dense_63/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
sequential_28/dense_63/Relu?
!sequential_28/dropout_93/IdentityIdentity)sequential_28/dense_63/Relu:activations:0*
T0*,
_output_shapes
:??????????2#
!sequential_28/dropout_93/Identity?
/sequential_28/dense_64/Tensordot/ReadVariableOpReadVariableOp8sequential_28_dense_64_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential_28/dense_64/Tensordot/ReadVariableOp?
%sequential_28/dense_64/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_28/dense_64/Tensordot/axes?
%sequential_28/dense_64/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_28/dense_64/Tensordot/free?
&sequential_28/dense_64/Tensordot/ShapeShape*sequential_28/dropout_93/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_28/dense_64/Tensordot/Shape?
.sequential_28/dense_64/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_28/dense_64/Tensordot/GatherV2/axis?
)sequential_28/dense_64/Tensordot/GatherV2GatherV2/sequential_28/dense_64/Tensordot/Shape:output:0.sequential_28/dense_64/Tensordot/free:output:07sequential_28/dense_64/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_28/dense_64/Tensordot/GatherV2?
0sequential_28/dense_64/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_28/dense_64/Tensordot/GatherV2_1/axis?
+sequential_28/dense_64/Tensordot/GatherV2_1GatherV2/sequential_28/dense_64/Tensordot/Shape:output:0.sequential_28/dense_64/Tensordot/axes:output:09sequential_28/dense_64/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_28/dense_64/Tensordot/GatherV2_1?
&sequential_28/dense_64/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_28/dense_64/Tensordot/Const?
%sequential_28/dense_64/Tensordot/ProdProd2sequential_28/dense_64/Tensordot/GatherV2:output:0/sequential_28/dense_64/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_28/dense_64/Tensordot/Prod?
(sequential_28/dense_64/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_28/dense_64/Tensordot/Const_1?
'sequential_28/dense_64/Tensordot/Prod_1Prod4sequential_28/dense_64/Tensordot/GatherV2_1:output:01sequential_28/dense_64/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_28/dense_64/Tensordot/Prod_1?
,sequential_28/dense_64/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_28/dense_64/Tensordot/concat/axis?
'sequential_28/dense_64/Tensordot/concatConcatV2.sequential_28/dense_64/Tensordot/free:output:0.sequential_28/dense_64/Tensordot/axes:output:05sequential_28/dense_64/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_28/dense_64/Tensordot/concat?
&sequential_28/dense_64/Tensordot/stackPack.sequential_28/dense_64/Tensordot/Prod:output:00sequential_28/dense_64/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_28/dense_64/Tensordot/stack?
*sequential_28/dense_64/Tensordot/transpose	Transpose*sequential_28/dropout_93/Identity:output:00sequential_28/dense_64/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2,
*sequential_28/dense_64/Tensordot/transpose?
(sequential_28/dense_64/Tensordot/ReshapeReshape.sequential_28/dense_64/Tensordot/transpose:y:0/sequential_28/dense_64/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_28/dense_64/Tensordot/Reshape?
'sequential_28/dense_64/Tensordot/MatMulMatMul1sequential_28/dense_64/Tensordot/Reshape:output:07sequential_28/dense_64/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_28/dense_64/Tensordot/MatMul?
(sequential_28/dense_64/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2*
(sequential_28/dense_64/Tensordot/Const_2?
.sequential_28/dense_64/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_28/dense_64/Tensordot/concat_1/axis?
)sequential_28/dense_64/Tensordot/concat_1ConcatV22sequential_28/dense_64/Tensordot/GatherV2:output:01sequential_28/dense_64/Tensordot/Const_2:output:07sequential_28/dense_64/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_28/dense_64/Tensordot/concat_1?
 sequential_28/dense_64/TensordotReshape1sequential_28/dense_64/Tensordot/MatMul:product:02sequential_28/dense_64/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_28/dense_64/Tensordot?
-sequential_28/dense_64/BiasAdd/ReadVariableOpReadVariableOp6sequential_28_dense_64_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_28/dense_64/BiasAdd/ReadVariableOp?
sequential_28/dense_64/BiasAddBiasAdd)sequential_28/dense_64/Tensordot:output:05sequential_28/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2 
sequential_28/dense_64/BiasAdd?
sequential_28/dense_64/ReluRelu'sequential_28/dense_64/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
sequential_28/dense_64/Relu?
!sequential_28/dropout_94/IdentityIdentity)sequential_28/dense_64/Relu:activations:0*
T0*,
_output_shapes
:??????????2#
!sequential_28/dropout_94/Identity?
/sequential_28/dense_65/Tensordot/ReadVariableOpReadVariableOp8sequential_28_dense_65_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_28/dense_65/Tensordot/ReadVariableOp?
%sequential_28/dense_65/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_28/dense_65/Tensordot/axes?
%sequential_28/dense_65/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_28/dense_65/Tensordot/free?
&sequential_28/dense_65/Tensordot/ShapeShape*sequential_28/dropout_94/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_28/dense_65/Tensordot/Shape?
.sequential_28/dense_65/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_28/dense_65/Tensordot/GatherV2/axis?
)sequential_28/dense_65/Tensordot/GatherV2GatherV2/sequential_28/dense_65/Tensordot/Shape:output:0.sequential_28/dense_65/Tensordot/free:output:07sequential_28/dense_65/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_28/dense_65/Tensordot/GatherV2?
0sequential_28/dense_65/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_28/dense_65/Tensordot/GatherV2_1/axis?
+sequential_28/dense_65/Tensordot/GatherV2_1GatherV2/sequential_28/dense_65/Tensordot/Shape:output:0.sequential_28/dense_65/Tensordot/axes:output:09sequential_28/dense_65/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_28/dense_65/Tensordot/GatherV2_1?
&sequential_28/dense_65/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_28/dense_65/Tensordot/Const?
%sequential_28/dense_65/Tensordot/ProdProd2sequential_28/dense_65/Tensordot/GatherV2:output:0/sequential_28/dense_65/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_28/dense_65/Tensordot/Prod?
(sequential_28/dense_65/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_28/dense_65/Tensordot/Const_1?
'sequential_28/dense_65/Tensordot/Prod_1Prod4sequential_28/dense_65/Tensordot/GatherV2_1:output:01sequential_28/dense_65/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_28/dense_65/Tensordot/Prod_1?
,sequential_28/dense_65/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_28/dense_65/Tensordot/concat/axis?
'sequential_28/dense_65/Tensordot/concatConcatV2.sequential_28/dense_65/Tensordot/free:output:0.sequential_28/dense_65/Tensordot/axes:output:05sequential_28/dense_65/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_28/dense_65/Tensordot/concat?
&sequential_28/dense_65/Tensordot/stackPack.sequential_28/dense_65/Tensordot/Prod:output:00sequential_28/dense_65/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_28/dense_65/Tensordot/stack?
*sequential_28/dense_65/Tensordot/transpose	Transpose*sequential_28/dropout_94/Identity:output:00sequential_28/dense_65/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2,
*sequential_28/dense_65/Tensordot/transpose?
(sequential_28/dense_65/Tensordot/ReshapeReshape.sequential_28/dense_65/Tensordot/transpose:y:0/sequential_28/dense_65/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_28/dense_65/Tensordot/Reshape?
'sequential_28/dense_65/Tensordot/MatMulMatMul1sequential_28/dense_65/Tensordot/Reshape:output:07sequential_28/dense_65/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'sequential_28/dense_65/Tensordot/MatMul?
(sequential_28/dense_65/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_28/dense_65/Tensordot/Const_2?
.sequential_28/dense_65/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_28/dense_65/Tensordot/concat_1/axis?
)sequential_28/dense_65/Tensordot/concat_1ConcatV22sequential_28/dense_65/Tensordot/GatherV2:output:01sequential_28/dense_65/Tensordot/Const_2:output:07sequential_28/dense_65/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_28/dense_65/Tensordot/concat_1?
 sequential_28/dense_65/TensordotReshape1sequential_28/dense_65/Tensordot/MatMul:product:02sequential_28/dense_65/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2"
 sequential_28/dense_65/Tensordot?
-sequential_28/dense_65/BiasAdd/ReadVariableOpReadVariableOp6sequential_28_dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_28/dense_65/BiasAdd/ReadVariableOp?
sequential_28/dense_65/BiasAddBiasAdd)sequential_28/dense_65/Tensordot:output:05sequential_28/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 
sequential_28/dense_65/BiasAdd?
IdentityIdentity'sequential_28/dense_65/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp.^sequential_28/dense_63/BiasAdd/ReadVariableOp0^sequential_28/dense_63/Tensordot/ReadVariableOp.^sequential_28/dense_64/BiasAdd/ReadVariableOp0^sequential_28/dense_64/Tensordot/ReadVariableOp.^sequential_28/dense_65/BiasAdd/ReadVariableOp0^sequential_28/dense_65/Tensordot/ReadVariableOp7^sequential_28/gru_56/gru_cell_56/MatMul/ReadVariableOp9^sequential_28/gru_56/gru_cell_56/MatMul_1/ReadVariableOp0^sequential_28/gru_56/gru_cell_56/ReadVariableOp^sequential_28/gru_56/while7^sequential_28/gru_57/gru_cell_57/MatMul/ReadVariableOp9^sequential_28/gru_57/gru_cell_57/MatMul_1/ReadVariableOp0^sequential_28/gru_57/gru_cell_57/ReadVariableOp^sequential_28/gru_57/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2^
-sequential_28/dense_63/BiasAdd/ReadVariableOp-sequential_28/dense_63/BiasAdd/ReadVariableOp2b
/sequential_28/dense_63/Tensordot/ReadVariableOp/sequential_28/dense_63/Tensordot/ReadVariableOp2^
-sequential_28/dense_64/BiasAdd/ReadVariableOp-sequential_28/dense_64/BiasAdd/ReadVariableOp2b
/sequential_28/dense_64/Tensordot/ReadVariableOp/sequential_28/dense_64/Tensordot/ReadVariableOp2^
-sequential_28/dense_65/BiasAdd/ReadVariableOp-sequential_28/dense_65/BiasAdd/ReadVariableOp2b
/sequential_28/dense_65/Tensordot/ReadVariableOp/sequential_28/dense_65/Tensordot/ReadVariableOp2p
6sequential_28/gru_56/gru_cell_56/MatMul/ReadVariableOp6sequential_28/gru_56/gru_cell_56/MatMul/ReadVariableOp2t
8sequential_28/gru_56/gru_cell_56/MatMul_1/ReadVariableOp8sequential_28/gru_56/gru_cell_56/MatMul_1/ReadVariableOp2b
/sequential_28/gru_56/gru_cell_56/ReadVariableOp/sequential_28/gru_56/gru_cell_56/ReadVariableOp28
sequential_28/gru_56/whilesequential_28/gru_56/while2p
6sequential_28/gru_57/gru_cell_57/MatMul/ReadVariableOp6sequential_28/gru_57/gru_cell_57/MatMul/ReadVariableOp2t
8sequential_28/gru_57/gru_cell_57/MatMul_1/ReadVariableOp8sequential_28/gru_57/gru_cell_57/MatMul_1/ReadVariableOp2b
/sequential_28/gru_57/gru_cell_57/ReadVariableOp/sequential_28/gru_57/gru_cell_57/ReadVariableOp28
sequential_28/gru_57/whilesequential_28/gru_57/while:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_56_input
?
?
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_3597489

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
?E
?
while_body_3600314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_56_readvariableop_resource_0:	?E
2while_gru_cell_56_matmul_readvariableop_resource_0:	?H
4while_gru_cell_56_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_56_readvariableop_resource:	?C
0while_gru_cell_56_matmul_readvariableop_resource:	?F
2while_gru_cell_56_matmul_1_readvariableop_resource:
????'while/gru_cell_56/MatMul/ReadVariableOp?)while/gru_cell_56/MatMul_1/ReadVariableOp? while/gru_cell_56/ReadVariableOp?
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
 while/gru_cell_56/ReadVariableOpReadVariableOp+while_gru_cell_56_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_56/ReadVariableOp?
while/gru_cell_56/unstackUnpack(while/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_56/unstack?
'while/gru_cell_56/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_56_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_56/MatMul/ReadVariableOp?
while/gru_cell_56/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul?
while/gru_cell_56/BiasAddBiasAdd"while/gru_cell_56/MatMul:product:0"while/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd?
!while/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_56/split/split_dim?
while/gru_cell_56/splitSplit*while/gru_cell_56/split/split_dim:output:0"while/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split?
)while/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_56_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_56/MatMul_1/ReadVariableOp?
while/gru_cell_56/MatMul_1MatMulwhile_placeholder_21while/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul_1?
while/gru_cell_56/BiasAdd_1BiasAdd$while/gru_cell_56/MatMul_1:product:0"while/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd_1?
while/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_56/Const?
#while/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_56/split_1/split_dim?
while/gru_cell_56/split_1SplitV$while/gru_cell_56/BiasAdd_1:output:0 while/gru_cell_56/Const:output:0,while/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split_1?
while/gru_cell_56/addAddV2 while/gru_cell_56/split:output:0"while/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add?
while/gru_cell_56/SigmoidSigmoidwhile/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid?
while/gru_cell_56/add_1AddV2 while/gru_cell_56/split:output:1"while/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_1?
while/gru_cell_56/Sigmoid_1Sigmoidwhile/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid_1?
while/gru_cell_56/mulMulwhile/gru_cell_56/Sigmoid_1:y:0"while/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul?
while/gru_cell_56/add_2AddV2 while/gru_cell_56/split:output:2while/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_2?
while/gru_cell_56/ReluReluwhile/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Relu?
while/gru_cell_56/mul_1Mulwhile/gru_cell_56/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_1w
while/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_56/sub/x?
while/gru_cell_56/subSub while/gru_cell_56/sub/x:output:0while/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/sub?
while/gru_cell_56/mul_2Mulwhile/gru_cell_56/sub:z:0$while/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_2?
while/gru_cell_56/add_3AddV2while/gru_cell_56/mul_1:z:0while/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_56/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_56/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_56/MatMul/ReadVariableOp*^while/gru_cell_56/MatMul_1/ReadVariableOp!^while/gru_cell_56/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_56_matmul_1_readvariableop_resource4while_gru_cell_56_matmul_1_readvariableop_resource_0"f
0while_gru_cell_56_matmul_readvariableop_resource2while_gru_cell_56_matmul_readvariableop_resource_0"X
)while_gru_cell_56_readvariableop_resource+while_gru_cell_56_readvariableop_resource_0")
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
'while/gru_cell_56/MatMul/ReadVariableOp'while/gru_cell_56/MatMul/ReadVariableOp2V
)while/gru_cell_56/MatMul_1/ReadVariableOp)while/gru_cell_56/MatMul_1/ReadVariableOp2D
 while/gru_cell_56/ReadVariableOp while/gru_cell_56/ReadVariableOp: 
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
while_body_3597552
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_57_3597574_0:	?/
while_gru_cell_57_3597576_0:
??/
while_gru_cell_57_3597578_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_57_3597574:	?-
while_gru_cell_57_3597576:
??-
while_gru_cell_57_3597578:
????)while/gru_cell_57/StatefulPartitionedCall?
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
)while/gru_cell_57/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_57_3597574_0while_gru_cell_57_3597576_0while_gru_cell_57_3597578_0*
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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_35974892+
)while/gru_cell_57/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_57/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_57/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"8
while_gru_cell_57_3597574while_gru_cell_57_3597574_0"8
while_gru_cell_57_3597576while_gru_cell_57_3597576_0"8
while_gru_cell_57_3597578while_gru_cell_57_3597578_0")
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
)while/gru_cell_57/StatefulPartitionedCall)while/gru_cell_57/StatefulPartitionedCall: 
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
while_body_3598709
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_56_readvariableop_resource_0:	?E
2while_gru_cell_56_matmul_readvariableop_resource_0:	?H
4while_gru_cell_56_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_56_readvariableop_resource:	?C
0while_gru_cell_56_matmul_readvariableop_resource:	?F
2while_gru_cell_56_matmul_1_readvariableop_resource:
????'while/gru_cell_56/MatMul/ReadVariableOp?)while/gru_cell_56/MatMul_1/ReadVariableOp? while/gru_cell_56/ReadVariableOp?
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
 while/gru_cell_56/ReadVariableOpReadVariableOp+while_gru_cell_56_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_56/ReadVariableOp?
while/gru_cell_56/unstackUnpack(while/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_56/unstack?
'while/gru_cell_56/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_56_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_56/MatMul/ReadVariableOp?
while/gru_cell_56/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul?
while/gru_cell_56/BiasAddBiasAdd"while/gru_cell_56/MatMul:product:0"while/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd?
!while/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_56/split/split_dim?
while/gru_cell_56/splitSplit*while/gru_cell_56/split/split_dim:output:0"while/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split?
)while/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_56_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_56/MatMul_1/ReadVariableOp?
while/gru_cell_56/MatMul_1MatMulwhile_placeholder_21while/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul_1?
while/gru_cell_56/BiasAdd_1BiasAdd$while/gru_cell_56/MatMul_1:product:0"while/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd_1?
while/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_56/Const?
#while/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_56/split_1/split_dim?
while/gru_cell_56/split_1SplitV$while/gru_cell_56/BiasAdd_1:output:0 while/gru_cell_56/Const:output:0,while/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split_1?
while/gru_cell_56/addAddV2 while/gru_cell_56/split:output:0"while/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add?
while/gru_cell_56/SigmoidSigmoidwhile/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid?
while/gru_cell_56/add_1AddV2 while/gru_cell_56/split:output:1"while/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_1?
while/gru_cell_56/Sigmoid_1Sigmoidwhile/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid_1?
while/gru_cell_56/mulMulwhile/gru_cell_56/Sigmoid_1:y:0"while/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul?
while/gru_cell_56/add_2AddV2 while/gru_cell_56/split:output:2while/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_2?
while/gru_cell_56/ReluReluwhile/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Relu?
while/gru_cell_56/mul_1Mulwhile/gru_cell_56/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_1w
while/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_56/sub/x?
while/gru_cell_56/subSub while/gru_cell_56/sub/x:output:0while/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/sub?
while/gru_cell_56/mul_2Mulwhile/gru_cell_56/sub:z:0$while/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_2?
while/gru_cell_56/add_3AddV2while/gru_cell_56/mul_1:z:0while/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_56/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_56/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_56/MatMul/ReadVariableOp*^while/gru_cell_56/MatMul_1/ReadVariableOp!^while/gru_cell_56/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_56_matmul_1_readvariableop_resource4while_gru_cell_56_matmul_1_readvariableop_resource_0"f
0while_gru_cell_56_matmul_readvariableop_resource2while_gru_cell_56_matmul_readvariableop_resource_0"X
)while_gru_cell_56_readvariableop_resource+while_gru_cell_56_readvariableop_resource_0")
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
'while/gru_cell_56/MatMul/ReadVariableOp'while/gru_cell_56/MatMul/ReadVariableOp2V
)while/gru_cell_56/MatMul_1/ReadVariableOp)while/gru_cell_56/MatMul_1/ReadVariableOp2D
 while/gru_cell_56/ReadVariableOp while/gru_cell_56/ReadVariableOp: 
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
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_3596923

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
?
?
while_cond_3600160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3600160___redundant_placeholder05
1while_while_cond_3600160___redundant_placeholder15
1while_while_cond_3600160___redundant_placeholder25
1while_while_cond_3600160___redundant_placeholder3
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
?P
?	
gru_57_while_body_3599707*
&gru_57_while_gru_57_while_loop_counter0
,gru_57_while_gru_57_while_maximum_iterations
gru_57_while_placeholder
gru_57_while_placeholder_1
gru_57_while_placeholder_2)
%gru_57_while_gru_57_strided_slice_1_0e
agru_57_while_tensorarrayv2read_tensorlistgetitem_gru_57_tensorarrayunstack_tensorlistfromtensor_0E
2gru_57_while_gru_cell_57_readvariableop_resource_0:	?M
9gru_57_while_gru_cell_57_matmul_readvariableop_resource_0:
??O
;gru_57_while_gru_cell_57_matmul_1_readvariableop_resource_0:
??
gru_57_while_identity
gru_57_while_identity_1
gru_57_while_identity_2
gru_57_while_identity_3
gru_57_while_identity_4'
#gru_57_while_gru_57_strided_slice_1c
_gru_57_while_tensorarrayv2read_tensorlistgetitem_gru_57_tensorarrayunstack_tensorlistfromtensorC
0gru_57_while_gru_cell_57_readvariableop_resource:	?K
7gru_57_while_gru_cell_57_matmul_readvariableop_resource:
??M
9gru_57_while_gru_cell_57_matmul_1_readvariableop_resource:
????.gru_57/while/gru_cell_57/MatMul/ReadVariableOp?0gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp?'gru_57/while/gru_cell_57/ReadVariableOp?
>gru_57/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_57/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_57/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_57_while_tensorarrayv2read_tensorlistgetitem_gru_57_tensorarrayunstack_tensorlistfromtensor_0gru_57_while_placeholderGgru_57/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0gru_57/while/TensorArrayV2Read/TensorListGetItem?
'gru_57/while/gru_cell_57/ReadVariableOpReadVariableOp2gru_57_while_gru_cell_57_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_57/while/gru_cell_57/ReadVariableOp?
 gru_57/while/gru_cell_57/unstackUnpack/gru_57/while/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_57/while/gru_cell_57/unstack?
.gru_57/while/gru_cell_57/MatMul/ReadVariableOpReadVariableOp9gru_57_while_gru_cell_57_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype020
.gru_57/while/gru_cell_57/MatMul/ReadVariableOp?
gru_57/while/gru_cell_57/MatMulMatMul7gru_57/while/TensorArrayV2Read/TensorListGetItem:item:06gru_57/while/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_57/while/gru_cell_57/MatMul?
 gru_57/while/gru_cell_57/BiasAddBiasAdd)gru_57/while/gru_cell_57/MatMul:product:0)gru_57/while/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_57/while/gru_cell_57/BiasAdd?
(gru_57/while/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_57/while/gru_cell_57/split/split_dim?
gru_57/while/gru_cell_57/splitSplit1gru_57/while/gru_cell_57/split/split_dim:output:0)gru_57/while/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_57/while/gru_cell_57/split?
0gru_57/while/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp;gru_57_while_gru_cell_57_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp?
!gru_57/while/gru_cell_57/MatMul_1MatMulgru_57_while_placeholder_28gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_57/while/gru_cell_57/MatMul_1?
"gru_57/while/gru_cell_57/BiasAdd_1BiasAdd+gru_57/while/gru_cell_57/MatMul_1:product:0)gru_57/while/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_57/while/gru_cell_57/BiasAdd_1?
gru_57/while/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2 
gru_57/while/gru_cell_57/Const?
*gru_57/while/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_57/while/gru_cell_57/split_1/split_dim?
 gru_57/while/gru_cell_57/split_1SplitV+gru_57/while/gru_cell_57/BiasAdd_1:output:0'gru_57/while/gru_cell_57/Const:output:03gru_57/while/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_57/while/gru_cell_57/split_1?
gru_57/while/gru_cell_57/addAddV2'gru_57/while/gru_cell_57/split:output:0)gru_57/while/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_57/while/gru_cell_57/add?
 gru_57/while/gru_cell_57/SigmoidSigmoid gru_57/while/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_57/while/gru_cell_57/Sigmoid?
gru_57/while/gru_cell_57/add_1AddV2'gru_57/while/gru_cell_57/split:output:1)gru_57/while/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_57/while/gru_cell_57/add_1?
"gru_57/while/gru_cell_57/Sigmoid_1Sigmoid"gru_57/while/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_57/while/gru_cell_57/Sigmoid_1?
gru_57/while/gru_cell_57/mulMul&gru_57/while/gru_cell_57/Sigmoid_1:y:0)gru_57/while/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_57/while/gru_cell_57/mul?
gru_57/while/gru_cell_57/add_2AddV2'gru_57/while/gru_cell_57/split:output:2 gru_57/while/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_57/while/gru_cell_57/add_2?
gru_57/while/gru_cell_57/ReluRelu"gru_57/while/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_57/while/gru_cell_57/Relu?
gru_57/while/gru_cell_57/mul_1Mul$gru_57/while/gru_cell_57/Sigmoid:y:0gru_57_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_57/while/gru_cell_57/mul_1?
gru_57/while/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_57/while/gru_cell_57/sub/x?
gru_57/while/gru_cell_57/subSub'gru_57/while/gru_cell_57/sub/x:output:0$gru_57/while/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_57/while/gru_cell_57/sub?
gru_57/while/gru_cell_57/mul_2Mul gru_57/while/gru_cell_57/sub:z:0+gru_57/while/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_57/while/gru_cell_57/mul_2?
gru_57/while/gru_cell_57/add_3AddV2"gru_57/while/gru_cell_57/mul_1:z:0"gru_57/while/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_57/while/gru_cell_57/add_3?
1gru_57/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_57_while_placeholder_1gru_57_while_placeholder"gru_57/while/gru_cell_57/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_57/while/TensorArrayV2Write/TensorListSetItemj
gru_57/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_57/while/add/y?
gru_57/while/addAddV2gru_57_while_placeholdergru_57/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_57/while/addn
gru_57/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_57/while/add_1/y?
gru_57/while/add_1AddV2&gru_57_while_gru_57_while_loop_countergru_57/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_57/while/add_1?
gru_57/while/IdentityIdentitygru_57/while/add_1:z:0^gru_57/while/NoOp*
T0*
_output_shapes
: 2
gru_57/while/Identity?
gru_57/while/Identity_1Identity,gru_57_while_gru_57_while_maximum_iterations^gru_57/while/NoOp*
T0*
_output_shapes
: 2
gru_57/while/Identity_1?
gru_57/while/Identity_2Identitygru_57/while/add:z:0^gru_57/while/NoOp*
T0*
_output_shapes
: 2
gru_57/while/Identity_2?
gru_57/while/Identity_3IdentityAgru_57/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_57/while/NoOp*
T0*
_output_shapes
: 2
gru_57/while/Identity_3?
gru_57/while/Identity_4Identity"gru_57/while/gru_cell_57/add_3:z:0^gru_57/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_57/while/Identity_4?
gru_57/while/NoOpNoOp/^gru_57/while/gru_cell_57/MatMul/ReadVariableOp1^gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp(^gru_57/while/gru_cell_57/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_57/while/NoOp"L
#gru_57_while_gru_57_strided_slice_1%gru_57_while_gru_57_strided_slice_1_0"x
9gru_57_while_gru_cell_57_matmul_1_readvariableop_resource;gru_57_while_gru_cell_57_matmul_1_readvariableop_resource_0"t
7gru_57_while_gru_cell_57_matmul_readvariableop_resource9gru_57_while_gru_cell_57_matmul_readvariableop_resource_0"f
0gru_57_while_gru_cell_57_readvariableop_resource2gru_57_while_gru_cell_57_readvariableop_resource_0"7
gru_57_while_identitygru_57/while/Identity:output:0";
gru_57_while_identity_1 gru_57/while/Identity_1:output:0";
gru_57_while_identity_2 gru_57/while/Identity_2:output:0";
gru_57_while_identity_3 gru_57/while/Identity_3:output:0";
gru_57_while_identity_4 gru_57/while/Identity_4:output:0"?
_gru_57_while_tensorarrayv2read_tensorlistgetitem_gru_57_tensorarrayunstack_tensorlistfromtensoragru_57_while_tensorarrayv2read_tensorlistgetitem_gru_57_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_57/while/gru_cell_57/MatMul/ReadVariableOp.gru_57/while/gru_cell_57/MatMul/ReadVariableOp2d
0gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp0gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp2R
'gru_57/while/gru_cell_57/ReadVariableOp'gru_57/while/gru_cell_57/ReadVariableOp: 
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
*__inference_dense_63_layer_call_fn_3601275

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
GPU 2J 8? *N
fIRG
E__inference_dense_63_layer_call_and_return_conditional_losses_35982142
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
?;
?
C__inference_gru_57_layer_call_and_return_conditional_losses_3597616

inputs&
gru_cell_57_3597540:	?'
gru_cell_57_3597542:
??'
gru_cell_57_3597544:
??
identity??#gru_cell_57/StatefulPartitionedCall?whileD
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
#gru_cell_57/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_57_3597540gru_cell_57_3597542gru_cell_57_3597544*
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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_35974892%
#gru_cell_57/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_57_3597540gru_cell_57_3597542gru_cell_57_3597544*
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
bodyR
while_body_3597552*
condR
while_cond_3597551*9
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
NoOpNoOp$^gru_cell_57/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#gru_cell_57/StatefulPartitionedCall#gru_cell_57/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_91_layer_call_and_return_conditional_losses_3598629

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
?
?
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_3597346

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
?
?
(__inference_gru_57_layer_call_fn_3600594
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
GPU 2J 8? *L
fGRE
C__inference_gru_57_layer_call_and_return_conditional_losses_35974232
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
?

?
-__inference_gru_cell_57_layer_call_fn_3601573

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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_35974892
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
?	
?
gru_56_while_cond_3599549*
&gru_56_while_gru_56_while_loop_counter0
,gru_56_while_gru_56_while_maximum_iterations
gru_56_while_placeholder
gru_56_while_placeholder_1
gru_56_while_placeholder_2,
(gru_56_while_less_gru_56_strided_slice_1C
?gru_56_while_gru_56_while_cond_3599549___redundant_placeholder0C
?gru_56_while_gru_56_while_cond_3599549___redundant_placeholder1C
?gru_56_while_gru_56_while_cond_3599549___redundant_placeholder2C
?gru_56_while_gru_56_while_cond_3599549___redundant_placeholder3
gru_56_while_identity
?
gru_56/while/LessLessgru_56_while_placeholder(gru_56_while_less_gru_56_strided_slice_1*
T0*
_output_shapes
: 2
gru_56/while/Lessr
gru_56/while/IdentityIdentitygru_56/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_56/while/Identity"7
gru_56_while_identitygru_56/while/Identity:output:0*(
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
e
,__inference_dropout_94_layer_call_fn_3601383

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
GPU 2J 8? *P
fKRI
G__inference_dropout_94_layer_call_and_return_conditional_losses_35983652
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
?"
?
while_body_3597359
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_57_3597381_0:	?/
while_gru_cell_57_3597383_0:
??/
while_gru_cell_57_3597385_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_57_3597381:	?-
while_gru_cell_57_3597383:
??-
while_gru_cell_57_3597385:
????)while/gru_cell_57/StatefulPartitionedCall?
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
)while/gru_cell_57/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_57_3597381_0while_gru_cell_57_3597383_0while_gru_cell_57_3597385_0*
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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_35973462+
)while/gru_cell_57/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_57/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_57/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"8
while_gru_cell_57_3597381while_gru_cell_57_3597381_0"8
while_gru_cell_57_3597383while_gru_cell_57_3597383_0"8
while_gru_cell_57_3597385while_gru_cell_57_3597385_0")
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
)while/gru_cell_57/StatefulPartitionedCall)while/gru_cell_57/StatefulPartitionedCall: 
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
?
gru_56_while_cond_3599163*
&gru_56_while_gru_56_while_loop_counter0
,gru_56_while_gru_56_while_maximum_iterations
gru_56_while_placeholder
gru_56_while_placeholder_1
gru_56_while_placeholder_2,
(gru_56_while_less_gru_56_strided_slice_1C
?gru_56_while_gru_56_while_cond_3599163___redundant_placeholder0C
?gru_56_while_gru_56_while_cond_3599163___redundant_placeholder1C
?gru_56_while_gru_56_while_cond_3599163___redundant_placeholder2C
?gru_56_while_gru_56_while_cond_3599163___redundant_placeholder3
gru_56_while_identity
?
gru_56/while/LessLessgru_56_while_placeholder(gru_56_while_less_gru_56_strided_slice_1*
T0*
_output_shapes
: 2
gru_56/while/Lessr
gru_56/while/IdentityIdentitygru_56/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_56/while/Identity"7
gru_56_while_identitygru_56/while/Identity:output:0*(
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
?f
?
'sequential_28_gru_56_while_body_3596388F
Bsequential_28_gru_56_while_sequential_28_gru_56_while_loop_counterL
Hsequential_28_gru_56_while_sequential_28_gru_56_while_maximum_iterations*
&sequential_28_gru_56_while_placeholder,
(sequential_28_gru_56_while_placeholder_1,
(sequential_28_gru_56_while_placeholder_2E
Asequential_28_gru_56_while_sequential_28_gru_56_strided_slice_1_0?
}sequential_28_gru_56_while_tensorarrayv2read_tensorlistgetitem_sequential_28_gru_56_tensorarrayunstack_tensorlistfromtensor_0S
@sequential_28_gru_56_while_gru_cell_56_readvariableop_resource_0:	?Z
Gsequential_28_gru_56_while_gru_cell_56_matmul_readvariableop_resource_0:	?]
Isequential_28_gru_56_while_gru_cell_56_matmul_1_readvariableop_resource_0:
??'
#sequential_28_gru_56_while_identity)
%sequential_28_gru_56_while_identity_1)
%sequential_28_gru_56_while_identity_2)
%sequential_28_gru_56_while_identity_3)
%sequential_28_gru_56_while_identity_4C
?sequential_28_gru_56_while_sequential_28_gru_56_strided_slice_1
{sequential_28_gru_56_while_tensorarrayv2read_tensorlistgetitem_sequential_28_gru_56_tensorarrayunstack_tensorlistfromtensorQ
>sequential_28_gru_56_while_gru_cell_56_readvariableop_resource:	?X
Esequential_28_gru_56_while_gru_cell_56_matmul_readvariableop_resource:	?[
Gsequential_28_gru_56_while_gru_cell_56_matmul_1_readvariableop_resource:
????<sequential_28/gru_56/while/gru_cell_56/MatMul/ReadVariableOp?>sequential_28/gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp?5sequential_28/gru_56/while/gru_cell_56/ReadVariableOp?
Lsequential_28/gru_56/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2N
Lsequential_28/gru_56/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_28/gru_56/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_28_gru_56_while_tensorarrayv2read_tensorlistgetitem_sequential_28_gru_56_tensorarrayunstack_tensorlistfromtensor_0&sequential_28_gru_56_while_placeholderUsequential_28/gru_56/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02@
>sequential_28/gru_56/while/TensorArrayV2Read/TensorListGetItem?
5sequential_28/gru_56/while/gru_cell_56/ReadVariableOpReadVariableOp@sequential_28_gru_56_while_gru_cell_56_readvariableop_resource_0*
_output_shapes
:	?*
dtype027
5sequential_28/gru_56/while/gru_cell_56/ReadVariableOp?
.sequential_28/gru_56/while/gru_cell_56/unstackUnpack=sequential_28/gru_56/while/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num20
.sequential_28/gru_56/while/gru_cell_56/unstack?
<sequential_28/gru_56/while/gru_cell_56/MatMul/ReadVariableOpReadVariableOpGsequential_28_gru_56_while_gru_cell_56_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02>
<sequential_28/gru_56/while/gru_cell_56/MatMul/ReadVariableOp?
-sequential_28/gru_56/while/gru_cell_56/MatMulMatMulEsequential_28/gru_56/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential_28/gru_56/while/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_28/gru_56/while/gru_cell_56/MatMul?
.sequential_28/gru_56/while/gru_cell_56/BiasAddBiasAdd7sequential_28/gru_56/while/gru_cell_56/MatMul:product:07sequential_28/gru_56/while/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????20
.sequential_28/gru_56/while/gru_cell_56/BiasAdd?
6sequential_28/gru_56/while/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_28/gru_56/while/gru_cell_56/split/split_dim?
,sequential_28/gru_56/while/gru_cell_56/splitSplit?sequential_28/gru_56/while/gru_cell_56/split/split_dim:output:07sequential_28/gru_56/while/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2.
,sequential_28/gru_56/while/gru_cell_56/split?
>sequential_28/gru_56/while/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOpIsequential_28_gru_56_while_gru_cell_56_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02@
>sequential_28/gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp?
/sequential_28/gru_56/while/gru_cell_56/MatMul_1MatMul(sequential_28_gru_56_while_placeholder_2Fsequential_28/gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_28/gru_56/while/gru_cell_56/MatMul_1?
0sequential_28/gru_56/while/gru_cell_56/BiasAdd_1BiasAdd9sequential_28/gru_56/while/gru_cell_56/MatMul_1:product:07sequential_28/gru_56/while/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????22
0sequential_28/gru_56/while/gru_cell_56/BiasAdd_1?
,sequential_28/gru_56/while/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2.
,sequential_28/gru_56/while/gru_cell_56/Const?
8sequential_28/gru_56/while/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8sequential_28/gru_56/while/gru_cell_56/split_1/split_dim?
.sequential_28/gru_56/while/gru_cell_56/split_1SplitV9sequential_28/gru_56/while/gru_cell_56/BiasAdd_1:output:05sequential_28/gru_56/while/gru_cell_56/Const:output:0Asequential_28/gru_56/while/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split20
.sequential_28/gru_56/while/gru_cell_56/split_1?
*sequential_28/gru_56/while/gru_cell_56/addAddV25sequential_28/gru_56/while/gru_cell_56/split:output:07sequential_28/gru_56/while/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2,
*sequential_28/gru_56/while/gru_cell_56/add?
.sequential_28/gru_56/while/gru_cell_56/SigmoidSigmoid.sequential_28/gru_56/while/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????20
.sequential_28/gru_56/while/gru_cell_56/Sigmoid?
,sequential_28/gru_56/while/gru_cell_56/add_1AddV25sequential_28/gru_56/while/gru_cell_56/split:output:17sequential_28/gru_56/while/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2.
,sequential_28/gru_56/while/gru_cell_56/add_1?
0sequential_28/gru_56/while/gru_cell_56/Sigmoid_1Sigmoid0sequential_28/gru_56/while/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????22
0sequential_28/gru_56/while/gru_cell_56/Sigmoid_1?
*sequential_28/gru_56/while/gru_cell_56/mulMul4sequential_28/gru_56/while/gru_cell_56/Sigmoid_1:y:07sequential_28/gru_56/while/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2,
*sequential_28/gru_56/while/gru_cell_56/mul?
,sequential_28/gru_56/while/gru_cell_56/add_2AddV25sequential_28/gru_56/while/gru_cell_56/split:output:2.sequential_28/gru_56/while/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2.
,sequential_28/gru_56/while/gru_cell_56/add_2?
+sequential_28/gru_56/while/gru_cell_56/ReluRelu0sequential_28/gru_56/while/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2-
+sequential_28/gru_56/while/gru_cell_56/Relu?
,sequential_28/gru_56/while/gru_cell_56/mul_1Mul2sequential_28/gru_56/while/gru_cell_56/Sigmoid:y:0(sequential_28_gru_56_while_placeholder_2*
T0*(
_output_shapes
:??????????2.
,sequential_28/gru_56/while/gru_cell_56/mul_1?
,sequential_28/gru_56/while/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,sequential_28/gru_56/while/gru_cell_56/sub/x?
*sequential_28/gru_56/while/gru_cell_56/subSub5sequential_28/gru_56/while/gru_cell_56/sub/x:output:02sequential_28/gru_56/while/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2,
*sequential_28/gru_56/while/gru_cell_56/sub?
,sequential_28/gru_56/while/gru_cell_56/mul_2Mul.sequential_28/gru_56/while/gru_cell_56/sub:z:09sequential_28/gru_56/while/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,sequential_28/gru_56/while/gru_cell_56/mul_2?
,sequential_28/gru_56/while/gru_cell_56/add_3AddV20sequential_28/gru_56/while/gru_cell_56/mul_1:z:00sequential_28/gru_56/while/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2.
,sequential_28/gru_56/while/gru_cell_56/add_3?
?sequential_28/gru_56/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_28_gru_56_while_placeholder_1&sequential_28_gru_56_while_placeholder0sequential_28/gru_56/while/gru_cell_56/add_3:z:0*
_output_shapes
: *
element_dtype02A
?sequential_28/gru_56/while/TensorArrayV2Write/TensorListSetItem?
 sequential_28/gru_56/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_28/gru_56/while/add/y?
sequential_28/gru_56/while/addAddV2&sequential_28_gru_56_while_placeholder)sequential_28/gru_56/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_28/gru_56/while/add?
"sequential_28/gru_56/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_28/gru_56/while/add_1/y?
 sequential_28/gru_56/while/add_1AddV2Bsequential_28_gru_56_while_sequential_28_gru_56_while_loop_counter+sequential_28/gru_56/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_28/gru_56/while/add_1?
#sequential_28/gru_56/while/IdentityIdentity$sequential_28/gru_56/while/add_1:z:0 ^sequential_28/gru_56/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_28/gru_56/while/Identity?
%sequential_28/gru_56/while/Identity_1IdentityHsequential_28_gru_56_while_sequential_28_gru_56_while_maximum_iterations ^sequential_28/gru_56/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_28/gru_56/while/Identity_1?
%sequential_28/gru_56/while/Identity_2Identity"sequential_28/gru_56/while/add:z:0 ^sequential_28/gru_56/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_28/gru_56/while/Identity_2?
%sequential_28/gru_56/while/Identity_3IdentityOsequential_28/gru_56/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_28/gru_56/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_28/gru_56/while/Identity_3?
%sequential_28/gru_56/while/Identity_4Identity0sequential_28/gru_56/while/gru_cell_56/add_3:z:0 ^sequential_28/gru_56/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_28/gru_56/while/Identity_4?
sequential_28/gru_56/while/NoOpNoOp=^sequential_28/gru_56/while/gru_cell_56/MatMul/ReadVariableOp?^sequential_28/gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp6^sequential_28/gru_56/while/gru_cell_56/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_28/gru_56/while/NoOp"?
Gsequential_28_gru_56_while_gru_cell_56_matmul_1_readvariableop_resourceIsequential_28_gru_56_while_gru_cell_56_matmul_1_readvariableop_resource_0"?
Esequential_28_gru_56_while_gru_cell_56_matmul_readvariableop_resourceGsequential_28_gru_56_while_gru_cell_56_matmul_readvariableop_resource_0"?
>sequential_28_gru_56_while_gru_cell_56_readvariableop_resource@sequential_28_gru_56_while_gru_cell_56_readvariableop_resource_0"S
#sequential_28_gru_56_while_identity,sequential_28/gru_56/while/Identity:output:0"W
%sequential_28_gru_56_while_identity_1.sequential_28/gru_56/while/Identity_1:output:0"W
%sequential_28_gru_56_while_identity_2.sequential_28/gru_56/while/Identity_2:output:0"W
%sequential_28_gru_56_while_identity_3.sequential_28/gru_56/while/Identity_3:output:0"W
%sequential_28_gru_56_while_identity_4.sequential_28/gru_56/while/Identity_4:output:0"?
?sequential_28_gru_56_while_sequential_28_gru_56_strided_slice_1Asequential_28_gru_56_while_sequential_28_gru_56_strided_slice_1_0"?
{sequential_28_gru_56_while_tensorarrayv2read_tensorlistgetitem_sequential_28_gru_56_tensorarrayunstack_tensorlistfromtensor}sequential_28_gru_56_while_tensorarrayv2read_tensorlistgetitem_sequential_28_gru_56_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2|
<sequential_28/gru_56/while/gru_cell_56/MatMul/ReadVariableOp<sequential_28/gru_56/while/gru_cell_56/MatMul/ReadVariableOp2?
>sequential_28/gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp>sequential_28/gru_56/while/gru_cell_56/MatMul_1/ReadVariableOp2n
5sequential_28/gru_56/while/gru_cell_56/ReadVariableOp5sequential_28/gru_56/while/gru_cell_56/ReadVariableOp: 
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
while_body_3600008
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_56_readvariableop_resource_0:	?E
2while_gru_cell_56_matmul_readvariableop_resource_0:	?H
4while_gru_cell_56_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_56_readvariableop_resource:	?C
0while_gru_cell_56_matmul_readvariableop_resource:	?F
2while_gru_cell_56_matmul_1_readvariableop_resource:
????'while/gru_cell_56/MatMul/ReadVariableOp?)while/gru_cell_56/MatMul_1/ReadVariableOp? while/gru_cell_56/ReadVariableOp?
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
 while/gru_cell_56/ReadVariableOpReadVariableOp+while_gru_cell_56_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_56/ReadVariableOp?
while/gru_cell_56/unstackUnpack(while/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_56/unstack?
'while/gru_cell_56/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_56_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_56/MatMul/ReadVariableOp?
while/gru_cell_56/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul?
while/gru_cell_56/BiasAddBiasAdd"while/gru_cell_56/MatMul:product:0"while/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd?
!while/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_56/split/split_dim?
while/gru_cell_56/splitSplit*while/gru_cell_56/split/split_dim:output:0"while/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split?
)while/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_56_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_56/MatMul_1/ReadVariableOp?
while/gru_cell_56/MatMul_1MatMulwhile_placeholder_21while/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul_1?
while/gru_cell_56/BiasAdd_1BiasAdd$while/gru_cell_56/MatMul_1:product:0"while/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd_1?
while/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_56/Const?
#while/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_56/split_1/split_dim?
while/gru_cell_56/split_1SplitV$while/gru_cell_56/BiasAdd_1:output:0 while/gru_cell_56/Const:output:0,while/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split_1?
while/gru_cell_56/addAddV2 while/gru_cell_56/split:output:0"while/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add?
while/gru_cell_56/SigmoidSigmoidwhile/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid?
while/gru_cell_56/add_1AddV2 while/gru_cell_56/split:output:1"while/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_1?
while/gru_cell_56/Sigmoid_1Sigmoidwhile/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid_1?
while/gru_cell_56/mulMulwhile/gru_cell_56/Sigmoid_1:y:0"while/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul?
while/gru_cell_56/add_2AddV2 while/gru_cell_56/split:output:2while/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_2?
while/gru_cell_56/ReluReluwhile/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Relu?
while/gru_cell_56/mul_1Mulwhile/gru_cell_56/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_1w
while/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_56/sub/x?
while/gru_cell_56/subSub while/gru_cell_56/sub/x:output:0while/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/sub?
while/gru_cell_56/mul_2Mulwhile/gru_cell_56/sub:z:0$while/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_2?
while/gru_cell_56/add_3AddV2while/gru_cell_56/mul_1:z:0while/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_56/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_56/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_56/MatMul/ReadVariableOp*^while/gru_cell_56/MatMul_1/ReadVariableOp!^while/gru_cell_56/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_56_matmul_1_readvariableop_resource4while_gru_cell_56_matmul_1_readvariableop_resource_0"f
0while_gru_cell_56_matmul_readvariableop_resource2while_gru_cell_56_matmul_readvariableop_resource_0"X
)while_gru_cell_56_readvariableop_resource+while_gru_cell_56_readvariableop_resource_0")
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
'while/gru_cell_56/MatMul/ReadVariableOp'while/gru_cell_56/MatMul/ReadVariableOp2V
)while/gru_cell_56/MatMul_1/ReadVariableOp)while/gru_cell_56/MatMul_1/ReadVariableOp2D
 while/gru_cell_56/ReadVariableOp while/gru_cell_56/ReadVariableOp: 
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
while_body_3600467
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_56_readvariableop_resource_0:	?E
2while_gru_cell_56_matmul_readvariableop_resource_0:	?H
4while_gru_cell_56_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_56_readvariableop_resource:	?C
0while_gru_cell_56_matmul_readvariableop_resource:	?F
2while_gru_cell_56_matmul_1_readvariableop_resource:
????'while/gru_cell_56/MatMul/ReadVariableOp?)while/gru_cell_56/MatMul_1/ReadVariableOp? while/gru_cell_56/ReadVariableOp?
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
 while/gru_cell_56/ReadVariableOpReadVariableOp+while_gru_cell_56_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_56/ReadVariableOp?
while/gru_cell_56/unstackUnpack(while/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_56/unstack?
'while/gru_cell_56/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_56_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_56/MatMul/ReadVariableOp?
while/gru_cell_56/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul?
while/gru_cell_56/BiasAddBiasAdd"while/gru_cell_56/MatMul:product:0"while/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd?
!while/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_56/split/split_dim?
while/gru_cell_56/splitSplit*while/gru_cell_56/split/split_dim:output:0"while/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split?
)while/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_56_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_56/MatMul_1/ReadVariableOp?
while/gru_cell_56/MatMul_1MatMulwhile_placeholder_21while/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul_1?
while/gru_cell_56/BiasAdd_1BiasAdd$while/gru_cell_56/MatMul_1:product:0"while/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd_1?
while/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_56/Const?
#while/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_56/split_1/split_dim?
while/gru_cell_56/split_1SplitV$while/gru_cell_56/BiasAdd_1:output:0 while/gru_cell_56/Const:output:0,while/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split_1?
while/gru_cell_56/addAddV2 while/gru_cell_56/split:output:0"while/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add?
while/gru_cell_56/SigmoidSigmoidwhile/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid?
while/gru_cell_56/add_1AddV2 while/gru_cell_56/split:output:1"while/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_1?
while/gru_cell_56/Sigmoid_1Sigmoidwhile/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid_1?
while/gru_cell_56/mulMulwhile/gru_cell_56/Sigmoid_1:y:0"while/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul?
while/gru_cell_56/add_2AddV2 while/gru_cell_56/split:output:2while/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_2?
while/gru_cell_56/ReluReluwhile/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Relu?
while/gru_cell_56/mul_1Mulwhile/gru_cell_56/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_1w
while/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_56/sub/x?
while/gru_cell_56/subSub while/gru_cell_56/sub/x:output:0while/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/sub?
while/gru_cell_56/mul_2Mulwhile/gru_cell_56/sub:z:0$while/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_2?
while/gru_cell_56/add_3AddV2while/gru_cell_56/mul_1:z:0while/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_56/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_56/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_56/MatMul/ReadVariableOp*^while/gru_cell_56/MatMul_1/ReadVariableOp!^while/gru_cell_56/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_56_matmul_1_readvariableop_resource4while_gru_cell_56_matmul_1_readvariableop_resource_0"f
0while_gru_cell_56_matmul_readvariableop_resource2while_gru_cell_56_matmul_readvariableop_resource_0"X
)while_gru_cell_56_readvariableop_resource+while_gru_cell_56_readvariableop_resource_0")
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
'while/gru_cell_56/MatMul/ReadVariableOp'while/gru_cell_56/MatMul/ReadVariableOp2V
)while/gru_cell_56/MatMul_1/ReadVariableOp)while/gru_cell_56/MatMul_1/ReadVariableOp2D
 while/gru_cell_56/ReadVariableOp while/gru_cell_56/ReadVariableOp: 
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
?f
?
'sequential_28_gru_57_while_body_3596538F
Bsequential_28_gru_57_while_sequential_28_gru_57_while_loop_counterL
Hsequential_28_gru_57_while_sequential_28_gru_57_while_maximum_iterations*
&sequential_28_gru_57_while_placeholder,
(sequential_28_gru_57_while_placeholder_1,
(sequential_28_gru_57_while_placeholder_2E
Asequential_28_gru_57_while_sequential_28_gru_57_strided_slice_1_0?
}sequential_28_gru_57_while_tensorarrayv2read_tensorlistgetitem_sequential_28_gru_57_tensorarrayunstack_tensorlistfromtensor_0S
@sequential_28_gru_57_while_gru_cell_57_readvariableop_resource_0:	?[
Gsequential_28_gru_57_while_gru_cell_57_matmul_readvariableop_resource_0:
??]
Isequential_28_gru_57_while_gru_cell_57_matmul_1_readvariableop_resource_0:
??'
#sequential_28_gru_57_while_identity)
%sequential_28_gru_57_while_identity_1)
%sequential_28_gru_57_while_identity_2)
%sequential_28_gru_57_while_identity_3)
%sequential_28_gru_57_while_identity_4C
?sequential_28_gru_57_while_sequential_28_gru_57_strided_slice_1
{sequential_28_gru_57_while_tensorarrayv2read_tensorlistgetitem_sequential_28_gru_57_tensorarrayunstack_tensorlistfromtensorQ
>sequential_28_gru_57_while_gru_cell_57_readvariableop_resource:	?Y
Esequential_28_gru_57_while_gru_cell_57_matmul_readvariableop_resource:
??[
Gsequential_28_gru_57_while_gru_cell_57_matmul_1_readvariableop_resource:
????<sequential_28/gru_57/while/gru_cell_57/MatMul/ReadVariableOp?>sequential_28/gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp?5sequential_28/gru_57/while/gru_cell_57/ReadVariableOp?
Lsequential_28/gru_57/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2N
Lsequential_28/gru_57/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_28/gru_57/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_28_gru_57_while_tensorarrayv2read_tensorlistgetitem_sequential_28_gru_57_tensorarrayunstack_tensorlistfromtensor_0&sequential_28_gru_57_while_placeholderUsequential_28/gru_57/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02@
>sequential_28/gru_57/while/TensorArrayV2Read/TensorListGetItem?
5sequential_28/gru_57/while/gru_cell_57/ReadVariableOpReadVariableOp@sequential_28_gru_57_while_gru_cell_57_readvariableop_resource_0*
_output_shapes
:	?*
dtype027
5sequential_28/gru_57/while/gru_cell_57/ReadVariableOp?
.sequential_28/gru_57/while/gru_cell_57/unstackUnpack=sequential_28/gru_57/while/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num20
.sequential_28/gru_57/while/gru_cell_57/unstack?
<sequential_28/gru_57/while/gru_cell_57/MatMul/ReadVariableOpReadVariableOpGsequential_28_gru_57_while_gru_cell_57_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02>
<sequential_28/gru_57/while/gru_cell_57/MatMul/ReadVariableOp?
-sequential_28/gru_57/while/gru_cell_57/MatMulMatMulEsequential_28/gru_57/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential_28/gru_57/while/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_28/gru_57/while/gru_cell_57/MatMul?
.sequential_28/gru_57/while/gru_cell_57/BiasAddBiasAdd7sequential_28/gru_57/while/gru_cell_57/MatMul:product:07sequential_28/gru_57/while/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????20
.sequential_28/gru_57/while/gru_cell_57/BiasAdd?
6sequential_28/gru_57/while/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_28/gru_57/while/gru_cell_57/split/split_dim?
,sequential_28/gru_57/while/gru_cell_57/splitSplit?sequential_28/gru_57/while/gru_cell_57/split/split_dim:output:07sequential_28/gru_57/while/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2.
,sequential_28/gru_57/while/gru_cell_57/split?
>sequential_28/gru_57/while/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOpIsequential_28_gru_57_while_gru_cell_57_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02@
>sequential_28/gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp?
/sequential_28/gru_57/while/gru_cell_57/MatMul_1MatMul(sequential_28_gru_57_while_placeholder_2Fsequential_28/gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_28/gru_57/while/gru_cell_57/MatMul_1?
0sequential_28/gru_57/while/gru_cell_57/BiasAdd_1BiasAdd9sequential_28/gru_57/while/gru_cell_57/MatMul_1:product:07sequential_28/gru_57/while/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????22
0sequential_28/gru_57/while/gru_cell_57/BiasAdd_1?
,sequential_28/gru_57/while/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2.
,sequential_28/gru_57/while/gru_cell_57/Const?
8sequential_28/gru_57/while/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8sequential_28/gru_57/while/gru_cell_57/split_1/split_dim?
.sequential_28/gru_57/while/gru_cell_57/split_1SplitV9sequential_28/gru_57/while/gru_cell_57/BiasAdd_1:output:05sequential_28/gru_57/while/gru_cell_57/Const:output:0Asequential_28/gru_57/while/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split20
.sequential_28/gru_57/while/gru_cell_57/split_1?
*sequential_28/gru_57/while/gru_cell_57/addAddV25sequential_28/gru_57/while/gru_cell_57/split:output:07sequential_28/gru_57/while/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2,
*sequential_28/gru_57/while/gru_cell_57/add?
.sequential_28/gru_57/while/gru_cell_57/SigmoidSigmoid.sequential_28/gru_57/while/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????20
.sequential_28/gru_57/while/gru_cell_57/Sigmoid?
,sequential_28/gru_57/while/gru_cell_57/add_1AddV25sequential_28/gru_57/while/gru_cell_57/split:output:17sequential_28/gru_57/while/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2.
,sequential_28/gru_57/while/gru_cell_57/add_1?
0sequential_28/gru_57/while/gru_cell_57/Sigmoid_1Sigmoid0sequential_28/gru_57/while/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????22
0sequential_28/gru_57/while/gru_cell_57/Sigmoid_1?
*sequential_28/gru_57/while/gru_cell_57/mulMul4sequential_28/gru_57/while/gru_cell_57/Sigmoid_1:y:07sequential_28/gru_57/while/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2,
*sequential_28/gru_57/while/gru_cell_57/mul?
,sequential_28/gru_57/while/gru_cell_57/add_2AddV25sequential_28/gru_57/while/gru_cell_57/split:output:2.sequential_28/gru_57/while/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2.
,sequential_28/gru_57/while/gru_cell_57/add_2?
+sequential_28/gru_57/while/gru_cell_57/ReluRelu0sequential_28/gru_57/while/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2-
+sequential_28/gru_57/while/gru_cell_57/Relu?
,sequential_28/gru_57/while/gru_cell_57/mul_1Mul2sequential_28/gru_57/while/gru_cell_57/Sigmoid:y:0(sequential_28_gru_57_while_placeholder_2*
T0*(
_output_shapes
:??????????2.
,sequential_28/gru_57/while/gru_cell_57/mul_1?
,sequential_28/gru_57/while/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,sequential_28/gru_57/while/gru_cell_57/sub/x?
*sequential_28/gru_57/while/gru_cell_57/subSub5sequential_28/gru_57/while/gru_cell_57/sub/x:output:02sequential_28/gru_57/while/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2,
*sequential_28/gru_57/while/gru_cell_57/sub?
,sequential_28/gru_57/while/gru_cell_57/mul_2Mul.sequential_28/gru_57/while/gru_cell_57/sub:z:09sequential_28/gru_57/while/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,sequential_28/gru_57/while/gru_cell_57/mul_2?
,sequential_28/gru_57/while/gru_cell_57/add_3AddV20sequential_28/gru_57/while/gru_cell_57/mul_1:z:00sequential_28/gru_57/while/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2.
,sequential_28/gru_57/while/gru_cell_57/add_3?
?sequential_28/gru_57/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_28_gru_57_while_placeholder_1&sequential_28_gru_57_while_placeholder0sequential_28/gru_57/while/gru_cell_57/add_3:z:0*
_output_shapes
: *
element_dtype02A
?sequential_28/gru_57/while/TensorArrayV2Write/TensorListSetItem?
 sequential_28/gru_57/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_28/gru_57/while/add/y?
sequential_28/gru_57/while/addAddV2&sequential_28_gru_57_while_placeholder)sequential_28/gru_57/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_28/gru_57/while/add?
"sequential_28/gru_57/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_28/gru_57/while/add_1/y?
 sequential_28/gru_57/while/add_1AddV2Bsequential_28_gru_57_while_sequential_28_gru_57_while_loop_counter+sequential_28/gru_57/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_28/gru_57/while/add_1?
#sequential_28/gru_57/while/IdentityIdentity$sequential_28/gru_57/while/add_1:z:0 ^sequential_28/gru_57/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_28/gru_57/while/Identity?
%sequential_28/gru_57/while/Identity_1IdentityHsequential_28_gru_57_while_sequential_28_gru_57_while_maximum_iterations ^sequential_28/gru_57/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_28/gru_57/while/Identity_1?
%sequential_28/gru_57/while/Identity_2Identity"sequential_28/gru_57/while/add:z:0 ^sequential_28/gru_57/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_28/gru_57/while/Identity_2?
%sequential_28/gru_57/while/Identity_3IdentityOsequential_28/gru_57/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_28/gru_57/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_28/gru_57/while/Identity_3?
%sequential_28/gru_57/while/Identity_4Identity0sequential_28/gru_57/while/gru_cell_57/add_3:z:0 ^sequential_28/gru_57/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_28/gru_57/while/Identity_4?
sequential_28/gru_57/while/NoOpNoOp=^sequential_28/gru_57/while/gru_cell_57/MatMul/ReadVariableOp?^sequential_28/gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp6^sequential_28/gru_57/while/gru_cell_57/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_28/gru_57/while/NoOp"?
Gsequential_28_gru_57_while_gru_cell_57_matmul_1_readvariableop_resourceIsequential_28_gru_57_while_gru_cell_57_matmul_1_readvariableop_resource_0"?
Esequential_28_gru_57_while_gru_cell_57_matmul_readvariableop_resourceGsequential_28_gru_57_while_gru_cell_57_matmul_readvariableop_resource_0"?
>sequential_28_gru_57_while_gru_cell_57_readvariableop_resource@sequential_28_gru_57_while_gru_cell_57_readvariableop_resource_0"S
#sequential_28_gru_57_while_identity,sequential_28/gru_57/while/Identity:output:0"W
%sequential_28_gru_57_while_identity_1.sequential_28/gru_57/while/Identity_1:output:0"W
%sequential_28_gru_57_while_identity_2.sequential_28/gru_57/while/Identity_2:output:0"W
%sequential_28_gru_57_while_identity_3.sequential_28/gru_57/while/Identity_3:output:0"W
%sequential_28_gru_57_while_identity_4.sequential_28/gru_57/while/Identity_4:output:0"?
?sequential_28_gru_57_while_sequential_28_gru_57_strided_slice_1Asequential_28_gru_57_while_sequential_28_gru_57_strided_slice_1_0"?
{sequential_28_gru_57_while_tensorarrayv2read_tensorlistgetitem_sequential_28_gru_57_tensorarrayunstack_tensorlistfromtensor}sequential_28_gru_57_while_tensorarrayv2read_tensorlistgetitem_sequential_28_gru_57_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2|
<sequential_28/gru_57/while/gru_cell_57/MatMul/ReadVariableOp<sequential_28/gru_57/while/gru_cell_57/MatMul/ReadVariableOp2?
>sequential_28/gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp>sequential_28/gru_57/while/gru_cell_57/MatMul_1/ReadVariableOp2n
5sequential_28/gru_57/while/gru_cell_57/ReadVariableOp5sequential_28/gru_57/while/gru_cell_57/ReadVariableOp: 
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
E__inference_dense_65_layer_call_and_return_conditional_losses_3601439

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
?
e
G__inference_dropout_92_layer_call_and_return_conditional_losses_3601254

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
?1
?
J__inference_sequential_28_layer_call_and_return_conditional_losses_3599005
gru_56_input!
gru_56_3598971:	?!
gru_56_3598973:	?"
gru_56_3598975:
??!
gru_57_3598979:	?"
gru_57_3598981:
??"
gru_57_3598983:
??$
dense_63_3598987:
??
dense_63_3598989:	?$
dense_64_3598993:
??
dense_64_3598995:	?#
dense_65_3598999:	?
dense_65_3599001:
identity?? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall?"dropout_91/StatefulPartitionedCall?"dropout_92/StatefulPartitionedCall?"dropout_93/StatefulPartitionedCall?"dropout_94/StatefulPartitionedCall?gru_56/StatefulPartitionedCall?gru_57/StatefulPartitionedCall?
gru_56/StatefulPartitionedCallStatefulPartitionedCallgru_56_inputgru_56_3598971gru_56_3598973gru_56_3598975*
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
GPU 2J 8? *L
fGRE
C__inference_gru_56_layer_call_and_return_conditional_losses_35987982 
gru_56/StatefulPartitionedCall?
"dropout_91/StatefulPartitionedCallStatefulPartitionedCall'gru_56/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_35986292$
"dropout_91/StatefulPartitionedCall?
gru_57/StatefulPartitionedCallStatefulPartitionedCall+dropout_91/StatefulPartitionedCall:output:0gru_57_3598979gru_57_3598981gru_57_3598983*
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
GPU 2J 8? *L
fGRE
C__inference_gru_57_layer_call_and_return_conditional_losses_35986002 
gru_57/StatefulPartitionedCall?
"dropout_92/StatefulPartitionedCallStatefulPartitionedCall'gru_57/StatefulPartitionedCall:output:0#^dropout_91/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_35984312$
"dropout_92/StatefulPartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall+dropout_92/StatefulPartitionedCall:output:0dense_63_3598987dense_63_3598989*
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
GPU 2J 8? *N
fIRG
E__inference_dense_63_layer_call_and_return_conditional_losses_35982142"
 dense_63/StatefulPartitionedCall?
"dropout_93/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0#^dropout_92/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_93_layer_call_and_return_conditional_losses_35983982$
"dropout_93/StatefulPartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall+dropout_93/StatefulPartitionedCall:output:0dense_64_3598993dense_64_3598995*
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
GPU 2J 8? *N
fIRG
E__inference_dense_64_layer_call_and_return_conditional_losses_35982582"
 dense_64/StatefulPartitionedCall?
"dropout_94/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0#^dropout_93/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_94_layer_call_and_return_conditional_losses_35983652$
"dropout_94/StatefulPartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall+dropout_94/StatefulPartitionedCall:output:0dense_65_3598999dense_65_3599001*
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
GPU 2J 8? *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_35983012"
 dense_65/StatefulPartitionedCall?
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall#^dropout_91/StatefulPartitionedCall#^dropout_92/StatefulPartitionedCall#^dropout_93/StatefulPartitionedCall#^dropout_94/StatefulPartitionedCall^gru_56/StatefulPartitionedCall^gru_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2H
"dropout_91/StatefulPartitionedCall"dropout_91/StatefulPartitionedCall2H
"dropout_92/StatefulPartitionedCall"dropout_92/StatefulPartitionedCall2H
"dropout_93/StatefulPartitionedCall"dropout_93/StatefulPartitionedCall2H
"dropout_94/StatefulPartitionedCall"dropout_94/StatefulPartitionedCall2@
gru_56/StatefulPartitionedCallgru_56/StatefulPartitionedCall2@
gru_57/StatefulPartitionedCallgru_57/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_56_input
?
H
,__inference_dropout_91_layer_call_fn_3600561

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
GPU 2J 8? *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_35980142
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
?
while_cond_3600313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3600313___redundant_placeholder05
1while_while_cond_3600313___redundant_placeholder15
1while_while_cond_3600313___redundant_placeholder25
1while_while_cond_3600313___redundant_placeholder3
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
E__inference_dense_65_layer_call_and_return_conditional_losses_3598301

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
?
while_cond_3598510
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3598510___redundant_placeholder05
1while_while_cond_3598510___redundant_placeholder15
1while_while_cond_3598510___redundant_placeholder25
1while_while_cond_3598510___redundant_placeholder3
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
?X
?
C__inference_gru_56_layer_call_and_return_conditional_losses_3600556

inputs6
#gru_cell_56_readvariableop_resource:	?=
*gru_cell_56_matmul_readvariableop_resource:	?@
,gru_cell_56_matmul_1_readvariableop_resource:
??
identity??!gru_cell_56/MatMul/ReadVariableOp?#gru_cell_56/MatMul_1/ReadVariableOp?gru_cell_56/ReadVariableOp?whileD
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
gru_cell_56/ReadVariableOpReadVariableOp#gru_cell_56_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_56/ReadVariableOp?
gru_cell_56/unstackUnpack"gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_56/unstack?
!gru_cell_56/MatMul/ReadVariableOpReadVariableOp*gru_cell_56_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_56/MatMul/ReadVariableOp?
gru_cell_56/MatMulMatMulstrided_slice_2:output:0)gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul?
gru_cell_56/BiasAddBiasAddgru_cell_56/MatMul:product:0gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd?
gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split/split_dim?
gru_cell_56/splitSplit$gru_cell_56/split/split_dim:output:0gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split?
#gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_56_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_56/MatMul_1/ReadVariableOp?
gru_cell_56/MatMul_1MatMulzeros:output:0+gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul_1?
gru_cell_56/BiasAdd_1BiasAddgru_cell_56/MatMul_1:product:0gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd_1{
gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_56/Const?
gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split_1/split_dim?
gru_cell_56/split_1SplitVgru_cell_56/BiasAdd_1:output:0gru_cell_56/Const:output:0&gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split_1?
gru_cell_56/addAddV2gru_cell_56/split:output:0gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add}
gru_cell_56/SigmoidSigmoidgru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid?
gru_cell_56/add_1AddV2gru_cell_56/split:output:1gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_1?
gru_cell_56/Sigmoid_1Sigmoidgru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid_1?
gru_cell_56/mulMulgru_cell_56/Sigmoid_1:y:0gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul?
gru_cell_56/add_2AddV2gru_cell_56/split:output:2gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_2v
gru_cell_56/ReluRelugru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Relu?
gru_cell_56/mul_1Mulgru_cell_56/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_1k
gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_56/sub/x?
gru_cell_56/subSubgru_cell_56/sub/x:output:0gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/sub?
gru_cell_56/mul_2Mulgru_cell_56/sub:z:0gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_2?
gru_cell_56/add_3AddV2gru_cell_56/mul_1:z:0gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_56_readvariableop_resource*gru_cell_56_matmul_readvariableop_resource,gru_cell_56_matmul_1_readvariableop_resource*
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
bodyR
while_body_3600467*
condR
while_cond_3600466*9
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
NoOpNoOp"^gru_cell_56/MatMul/ReadVariableOp$^gru_cell_56/MatMul_1/ReadVariableOp^gru_cell_56/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_56/MatMul/ReadVariableOp!gru_cell_56/MatMul/ReadVariableOp2J
#gru_cell_56/MatMul_1/ReadVariableOp#gru_cell_56/MatMul_1/ReadVariableOp28
gru_cell_56/ReadVariableOpgru_cell_56/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?*
?
J__inference_sequential_28_layer_call_and_return_conditional_losses_3598308

inputs!
gru_56_3598002:	?!
gru_56_3598004:	?"
gru_56_3598006:
??!
gru_57_3598169:	?"
gru_57_3598171:
??"
gru_57_3598173:
??$
dense_63_3598215:
??
dense_63_3598217:	?$
dense_64_3598259:
??
dense_64_3598261:	?#
dense_65_3598302:	?
dense_65_3598304:
identity?? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall?gru_56/StatefulPartitionedCall?gru_57/StatefulPartitionedCall?
gru_56/StatefulPartitionedCallStatefulPartitionedCallinputsgru_56_3598002gru_56_3598004gru_56_3598006*
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
GPU 2J 8? *L
fGRE
C__inference_gru_56_layer_call_and_return_conditional_losses_35980012 
gru_56/StatefulPartitionedCall?
dropout_91/PartitionedCallPartitionedCall'gru_56/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_35980142
dropout_91/PartitionedCall?
gru_57/StatefulPartitionedCallStatefulPartitionedCall#dropout_91/PartitionedCall:output:0gru_57_3598169gru_57_3598171gru_57_3598173*
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
GPU 2J 8? *L
fGRE
C__inference_gru_57_layer_call_and_return_conditional_losses_35981682 
gru_57/StatefulPartitionedCall?
dropout_92/PartitionedCallPartitionedCall'gru_57/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_35981812
dropout_92/PartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall#dropout_92/PartitionedCall:output:0dense_63_3598215dense_63_3598217*
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
GPU 2J 8? *N
fIRG
E__inference_dense_63_layer_call_and_return_conditional_losses_35982142"
 dense_63/StatefulPartitionedCall?
dropout_93/PartitionedCallPartitionedCall)dense_63/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_93_layer_call_and_return_conditional_losses_35982252
dropout_93/PartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall#dropout_93/PartitionedCall:output:0dense_64_3598259dense_64_3598261*
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
GPU 2J 8? *N
fIRG
E__inference_dense_64_layer_call_and_return_conditional_losses_35982582"
 dense_64/StatefulPartitionedCall?
dropout_94/PartitionedCallPartitionedCall)dense_64/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_94_layer_call_and_return_conditional_losses_35982692
dropout_94/PartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall#dropout_94/PartitionedCall:output:0dense_65_3598302dense_65_3598304*
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
GPU 2J 8? *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_35983012"
 dense_65/StatefulPartitionedCall?
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall^gru_56/StatefulPartitionedCall^gru_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2@
gru_56/StatefulPartitionedCallgru_56/StatefulPartitionedCall2@
gru_57/StatefulPartitionedCallgru_57/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_3600007
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3600007___redundant_placeholder05
1while_while_cond_3600007___redundant_placeholder15
1while_while_cond_3600007___redundant_placeholder25
1while_while_cond_3600007___redundant_placeholder3
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
?
'sequential_28_gru_56_while_cond_3596387F
Bsequential_28_gru_56_while_sequential_28_gru_56_while_loop_counterL
Hsequential_28_gru_56_while_sequential_28_gru_56_while_maximum_iterations*
&sequential_28_gru_56_while_placeholder,
(sequential_28_gru_56_while_placeholder_1,
(sequential_28_gru_56_while_placeholder_2H
Dsequential_28_gru_56_while_less_sequential_28_gru_56_strided_slice_1_
[sequential_28_gru_56_while_sequential_28_gru_56_while_cond_3596387___redundant_placeholder0_
[sequential_28_gru_56_while_sequential_28_gru_56_while_cond_3596387___redundant_placeholder1_
[sequential_28_gru_56_while_sequential_28_gru_56_while_cond_3596387___redundant_placeholder2_
[sequential_28_gru_56_while_sequential_28_gru_56_while_cond_3596387___redundant_placeholder3'
#sequential_28_gru_56_while_identity
?
sequential_28/gru_56/while/LessLess&sequential_28_gru_56_while_placeholderDsequential_28_gru_56_while_less_sequential_28_gru_56_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_28/gru_56/while/Less?
#sequential_28/gru_56/while/IdentityIdentity#sequential_28/gru_56/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_28/gru_56/while/Identity"S
#sequential_28_gru_56_while_identity,sequential_28/gru_56/while/Identity:output:0*(
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
??
?

J__inference_sequential_28_layer_call_and_return_conditional_losses_3599486

inputs=
*gru_56_gru_cell_56_readvariableop_resource:	?D
1gru_56_gru_cell_56_matmul_readvariableop_resource:	?G
3gru_56_gru_cell_56_matmul_1_readvariableop_resource:
??=
*gru_57_gru_cell_57_readvariableop_resource:	?E
1gru_57_gru_cell_57_matmul_readvariableop_resource:
??G
3gru_57_gru_cell_57_matmul_1_readvariableop_resource:
??>
*dense_63_tensordot_readvariableop_resource:
??7
(dense_63_biasadd_readvariableop_resource:	?>
*dense_64_tensordot_readvariableop_resource:
??7
(dense_64_biasadd_readvariableop_resource:	?=
*dense_65_tensordot_readvariableop_resource:	?6
(dense_65_biasadd_readvariableop_resource:
identity??dense_63/BiasAdd/ReadVariableOp?!dense_63/Tensordot/ReadVariableOp?dense_64/BiasAdd/ReadVariableOp?!dense_64/Tensordot/ReadVariableOp?dense_65/BiasAdd/ReadVariableOp?!dense_65/Tensordot/ReadVariableOp?(gru_56/gru_cell_56/MatMul/ReadVariableOp?*gru_56/gru_cell_56/MatMul_1/ReadVariableOp?!gru_56/gru_cell_56/ReadVariableOp?gru_56/while?(gru_57/gru_cell_57/MatMul/ReadVariableOp?*gru_57/gru_cell_57/MatMul_1/ReadVariableOp?!gru_57/gru_cell_57/ReadVariableOp?gru_57/whileR
gru_56/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_56/Shape?
gru_56/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_56/strided_slice/stack?
gru_56/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_56/strided_slice/stack_1?
gru_56/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_56/strided_slice/stack_2?
gru_56/strided_sliceStridedSlicegru_56/Shape:output:0#gru_56/strided_slice/stack:output:0%gru_56/strided_slice/stack_1:output:0%gru_56/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_56/strided_sliceq
gru_56/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_56/zeros/packed/1?
gru_56/zeros/packedPackgru_56/strided_slice:output:0gru_56/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_56/zeros/packedm
gru_56/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_56/zeros/Const?
gru_56/zerosFillgru_56/zeros/packed:output:0gru_56/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_56/zeros?
gru_56/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_56/transpose/perm?
gru_56/transpose	Transposeinputsgru_56/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
gru_56/transposed
gru_56/Shape_1Shapegru_56/transpose:y:0*
T0*
_output_shapes
:2
gru_56/Shape_1?
gru_56/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_56/strided_slice_1/stack?
gru_56/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_56/strided_slice_1/stack_1?
gru_56/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_56/strided_slice_1/stack_2?
gru_56/strided_slice_1StridedSlicegru_56/Shape_1:output:0%gru_56/strided_slice_1/stack:output:0'gru_56/strided_slice_1/stack_1:output:0'gru_56/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_56/strided_slice_1?
"gru_56/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_56/TensorArrayV2/element_shape?
gru_56/TensorArrayV2TensorListReserve+gru_56/TensorArrayV2/element_shape:output:0gru_56/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_56/TensorArrayV2?
<gru_56/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_56/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_56/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_56/transpose:y:0Egru_56/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_56/TensorArrayUnstack/TensorListFromTensor?
gru_56/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_56/strided_slice_2/stack?
gru_56/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_56/strided_slice_2/stack_1?
gru_56/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_56/strided_slice_2/stack_2?
gru_56/strided_slice_2StridedSlicegru_56/transpose:y:0%gru_56/strided_slice_2/stack:output:0'gru_56/strided_slice_2/stack_1:output:0'gru_56/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_56/strided_slice_2?
!gru_56/gru_cell_56/ReadVariableOpReadVariableOp*gru_56_gru_cell_56_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_56/gru_cell_56/ReadVariableOp?
gru_56/gru_cell_56/unstackUnpack)gru_56/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_56/gru_cell_56/unstack?
(gru_56/gru_cell_56/MatMul/ReadVariableOpReadVariableOp1gru_56_gru_cell_56_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_56/gru_cell_56/MatMul/ReadVariableOp?
gru_56/gru_cell_56/MatMulMatMulgru_56/strided_slice_2:output:00gru_56/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/MatMul?
gru_56/gru_cell_56/BiasAddBiasAdd#gru_56/gru_cell_56/MatMul:product:0#gru_56/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/BiasAdd?
"gru_56/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_56/gru_cell_56/split/split_dim?
gru_56/gru_cell_56/splitSplit+gru_56/gru_cell_56/split/split_dim:output:0#gru_56/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_56/gru_cell_56/split?
*gru_56/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp3gru_56_gru_cell_56_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_56/gru_cell_56/MatMul_1/ReadVariableOp?
gru_56/gru_cell_56/MatMul_1MatMulgru_56/zeros:output:02gru_56/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/MatMul_1?
gru_56/gru_cell_56/BiasAdd_1BiasAdd%gru_56/gru_cell_56/MatMul_1:product:0#gru_56/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/BiasAdd_1?
gru_56/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_56/gru_cell_56/Const?
$gru_56/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_56/gru_cell_56/split_1/split_dim?
gru_56/gru_cell_56/split_1SplitV%gru_56/gru_cell_56/BiasAdd_1:output:0!gru_56/gru_cell_56/Const:output:0-gru_56/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_56/gru_cell_56/split_1?
gru_56/gru_cell_56/addAddV2!gru_56/gru_cell_56/split:output:0#gru_56/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/add?
gru_56/gru_cell_56/SigmoidSigmoidgru_56/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/Sigmoid?
gru_56/gru_cell_56/add_1AddV2!gru_56/gru_cell_56/split:output:1#gru_56/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/add_1?
gru_56/gru_cell_56/Sigmoid_1Sigmoidgru_56/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/Sigmoid_1?
gru_56/gru_cell_56/mulMul gru_56/gru_cell_56/Sigmoid_1:y:0#gru_56/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/mul?
gru_56/gru_cell_56/add_2AddV2!gru_56/gru_cell_56/split:output:2gru_56/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/add_2?
gru_56/gru_cell_56/ReluRelugru_56/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/Relu?
gru_56/gru_cell_56/mul_1Mulgru_56/gru_cell_56/Sigmoid:y:0gru_56/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/mul_1y
gru_56/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_56/gru_cell_56/sub/x?
gru_56/gru_cell_56/subSub!gru_56/gru_cell_56/sub/x:output:0gru_56/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/sub?
gru_56/gru_cell_56/mul_2Mulgru_56/gru_cell_56/sub:z:0%gru_56/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/mul_2?
gru_56/gru_cell_56/add_3AddV2gru_56/gru_cell_56/mul_1:z:0gru_56/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_56/gru_cell_56/add_3?
$gru_56/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$gru_56/TensorArrayV2_1/element_shape?
gru_56/TensorArrayV2_1TensorListReserve-gru_56/TensorArrayV2_1/element_shape:output:0gru_56/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_56/TensorArrayV2_1\
gru_56/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_56/time?
gru_56/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_56/while/maximum_iterationsx
gru_56/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_56/while/loop_counter?
gru_56/whileWhile"gru_56/while/loop_counter:output:0(gru_56/while/maximum_iterations:output:0gru_56/time:output:0gru_56/TensorArrayV2_1:handle:0gru_56/zeros:output:0gru_56/strided_slice_1:output:0>gru_56/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_56_gru_cell_56_readvariableop_resource1gru_56_gru_cell_56_matmul_readvariableop_resource3gru_56_gru_cell_56_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *%
bodyR
gru_56_while_body_3599164*%
condR
gru_56_while_cond_3599163*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_56/while?
7gru_56/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7gru_56/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_56/TensorArrayV2Stack/TensorListStackTensorListStackgru_56/while:output:3@gru_56/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_56/TensorArrayV2Stack/TensorListStack?
gru_56/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_56/strided_slice_3/stack?
gru_56/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_56/strided_slice_3/stack_1?
gru_56/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_56/strided_slice_3/stack_2?
gru_56/strided_slice_3StridedSlice2gru_56/TensorArrayV2Stack/TensorListStack:tensor:0%gru_56/strided_slice_3/stack:output:0'gru_56/strided_slice_3/stack_1:output:0'gru_56/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_56/strided_slice_3?
gru_56/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_56/transpose_1/perm?
gru_56/transpose_1	Transpose2gru_56/TensorArrayV2Stack/TensorListStack:tensor:0 gru_56/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_56/transpose_1t
gru_56/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_56/runtime?
dropout_91/IdentityIdentitygru_56/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_91/Identityh
gru_57/ShapeShapedropout_91/Identity:output:0*
T0*
_output_shapes
:2
gru_57/Shape?
gru_57/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_57/strided_slice/stack?
gru_57/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_57/strided_slice/stack_1?
gru_57/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_57/strided_slice/stack_2?
gru_57/strided_sliceStridedSlicegru_57/Shape:output:0#gru_57/strided_slice/stack:output:0%gru_57/strided_slice/stack_1:output:0%gru_57/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_57/strided_sliceq
gru_57/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_57/zeros/packed/1?
gru_57/zeros/packedPackgru_57/strided_slice:output:0gru_57/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_57/zeros/packedm
gru_57/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_57/zeros/Const?
gru_57/zerosFillgru_57/zeros/packed:output:0gru_57/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_57/zeros?
gru_57/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_57/transpose/perm?
gru_57/transpose	Transposedropout_91/Identity:output:0gru_57/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_57/transposed
gru_57/Shape_1Shapegru_57/transpose:y:0*
T0*
_output_shapes
:2
gru_57/Shape_1?
gru_57/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_57/strided_slice_1/stack?
gru_57/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_57/strided_slice_1/stack_1?
gru_57/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_57/strided_slice_1/stack_2?
gru_57/strided_slice_1StridedSlicegru_57/Shape_1:output:0%gru_57/strided_slice_1/stack:output:0'gru_57/strided_slice_1/stack_1:output:0'gru_57/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_57/strided_slice_1?
"gru_57/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_57/TensorArrayV2/element_shape?
gru_57/TensorArrayV2TensorListReserve+gru_57/TensorArrayV2/element_shape:output:0gru_57/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_57/TensorArrayV2?
<gru_57/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_57/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_57/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_57/transpose:y:0Egru_57/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_57/TensorArrayUnstack/TensorListFromTensor?
gru_57/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_57/strided_slice_2/stack?
gru_57/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_57/strided_slice_2/stack_1?
gru_57/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_57/strided_slice_2/stack_2?
gru_57/strided_slice_2StridedSlicegru_57/transpose:y:0%gru_57/strided_slice_2/stack:output:0'gru_57/strided_slice_2/stack_1:output:0'gru_57/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_57/strided_slice_2?
!gru_57/gru_cell_57/ReadVariableOpReadVariableOp*gru_57_gru_cell_57_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_57/gru_cell_57/ReadVariableOp?
gru_57/gru_cell_57/unstackUnpack)gru_57/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_57/gru_cell_57/unstack?
(gru_57/gru_cell_57/MatMul/ReadVariableOpReadVariableOp1gru_57_gru_cell_57_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(gru_57/gru_cell_57/MatMul/ReadVariableOp?
gru_57/gru_cell_57/MatMulMatMulgru_57/strided_slice_2:output:00gru_57/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/MatMul?
gru_57/gru_cell_57/BiasAddBiasAdd#gru_57/gru_cell_57/MatMul:product:0#gru_57/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/BiasAdd?
"gru_57/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_57/gru_cell_57/split/split_dim?
gru_57/gru_cell_57/splitSplit+gru_57/gru_cell_57/split/split_dim:output:0#gru_57/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_57/gru_cell_57/split?
*gru_57/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp3gru_57_gru_cell_57_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_57/gru_cell_57/MatMul_1/ReadVariableOp?
gru_57/gru_cell_57/MatMul_1MatMulgru_57/zeros:output:02gru_57/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/MatMul_1?
gru_57/gru_cell_57/BiasAdd_1BiasAdd%gru_57/gru_cell_57/MatMul_1:product:0#gru_57/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/BiasAdd_1?
gru_57/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_57/gru_cell_57/Const?
$gru_57/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_57/gru_cell_57/split_1/split_dim?
gru_57/gru_cell_57/split_1SplitV%gru_57/gru_cell_57/BiasAdd_1:output:0!gru_57/gru_cell_57/Const:output:0-gru_57/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_57/gru_cell_57/split_1?
gru_57/gru_cell_57/addAddV2!gru_57/gru_cell_57/split:output:0#gru_57/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/add?
gru_57/gru_cell_57/SigmoidSigmoidgru_57/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/Sigmoid?
gru_57/gru_cell_57/add_1AddV2!gru_57/gru_cell_57/split:output:1#gru_57/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/add_1?
gru_57/gru_cell_57/Sigmoid_1Sigmoidgru_57/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/Sigmoid_1?
gru_57/gru_cell_57/mulMul gru_57/gru_cell_57/Sigmoid_1:y:0#gru_57/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/mul?
gru_57/gru_cell_57/add_2AddV2!gru_57/gru_cell_57/split:output:2gru_57/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/add_2?
gru_57/gru_cell_57/ReluRelugru_57/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/Relu?
gru_57/gru_cell_57/mul_1Mulgru_57/gru_cell_57/Sigmoid:y:0gru_57/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/mul_1y
gru_57/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_57/gru_cell_57/sub/x?
gru_57/gru_cell_57/subSub!gru_57/gru_cell_57/sub/x:output:0gru_57/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/sub?
gru_57/gru_cell_57/mul_2Mulgru_57/gru_cell_57/sub:z:0%gru_57/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/mul_2?
gru_57/gru_cell_57/add_3AddV2gru_57/gru_cell_57/mul_1:z:0gru_57/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_57/gru_cell_57/add_3?
$gru_57/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$gru_57/TensorArrayV2_1/element_shape?
gru_57/TensorArrayV2_1TensorListReserve-gru_57/TensorArrayV2_1/element_shape:output:0gru_57/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_57/TensorArrayV2_1\
gru_57/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_57/time?
gru_57/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_57/while/maximum_iterationsx
gru_57/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_57/while/loop_counter?
gru_57/whileWhile"gru_57/while/loop_counter:output:0(gru_57/while/maximum_iterations:output:0gru_57/time:output:0gru_57/TensorArrayV2_1:handle:0gru_57/zeros:output:0gru_57/strided_slice_1:output:0>gru_57/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_57_gru_cell_57_readvariableop_resource1gru_57_gru_cell_57_matmul_readvariableop_resource3gru_57_gru_cell_57_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *%
bodyR
gru_57_while_body_3599314*%
condR
gru_57_while_cond_3599313*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_57/while?
7gru_57/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7gru_57/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_57/TensorArrayV2Stack/TensorListStackTensorListStackgru_57/while:output:3@gru_57/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_57/TensorArrayV2Stack/TensorListStack?
gru_57/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_57/strided_slice_3/stack?
gru_57/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_57/strided_slice_3/stack_1?
gru_57/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_57/strided_slice_3/stack_2?
gru_57/strided_slice_3StridedSlice2gru_57/TensorArrayV2Stack/TensorListStack:tensor:0%gru_57/strided_slice_3/stack:output:0'gru_57/strided_slice_3/stack_1:output:0'gru_57/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_57/strided_slice_3?
gru_57/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_57/transpose_1/perm?
gru_57/transpose_1	Transpose2gru_57/TensorArrayV2Stack/TensorListStack:tensor:0 gru_57/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_57/transpose_1t
gru_57/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_57/runtime?
dropout_92/IdentityIdentitygru_57/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_92/Identity?
!dense_63/Tensordot/ReadVariableOpReadVariableOp*dense_63_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_63/Tensordot/ReadVariableOp|
dense_63/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_63/Tensordot/axes?
dense_63/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_63/Tensordot/free?
dense_63/Tensordot/ShapeShapedropout_92/Identity:output:0*
T0*
_output_shapes
:2
dense_63/Tensordot/Shape?
 dense_63/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_63/Tensordot/GatherV2/axis?
dense_63/Tensordot/GatherV2GatherV2!dense_63/Tensordot/Shape:output:0 dense_63/Tensordot/free:output:0)dense_63/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_63/Tensordot/GatherV2?
"dense_63/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_63/Tensordot/GatherV2_1/axis?
dense_63/Tensordot/GatherV2_1GatherV2!dense_63/Tensordot/Shape:output:0 dense_63/Tensordot/axes:output:0+dense_63/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_63/Tensordot/GatherV2_1~
dense_63/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_63/Tensordot/Const?
dense_63/Tensordot/ProdProd$dense_63/Tensordot/GatherV2:output:0!dense_63/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_63/Tensordot/Prod?
dense_63/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_63/Tensordot/Const_1?
dense_63/Tensordot/Prod_1Prod&dense_63/Tensordot/GatherV2_1:output:0#dense_63/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_63/Tensordot/Prod_1?
dense_63/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_63/Tensordot/concat/axis?
dense_63/Tensordot/concatConcatV2 dense_63/Tensordot/free:output:0 dense_63/Tensordot/axes:output:0'dense_63/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_63/Tensordot/concat?
dense_63/Tensordot/stackPack dense_63/Tensordot/Prod:output:0"dense_63/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_63/Tensordot/stack?
dense_63/Tensordot/transpose	Transposedropout_92/Identity:output:0"dense_63/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_63/Tensordot/transpose?
dense_63/Tensordot/ReshapeReshape dense_63/Tensordot/transpose:y:0!dense_63/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_63/Tensordot/Reshape?
dense_63/Tensordot/MatMulMatMul#dense_63/Tensordot/Reshape:output:0)dense_63/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_63/Tensordot/MatMul?
dense_63/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_63/Tensordot/Const_2?
 dense_63/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_63/Tensordot/concat_1/axis?
dense_63/Tensordot/concat_1ConcatV2$dense_63/Tensordot/GatherV2:output:0#dense_63/Tensordot/Const_2:output:0)dense_63/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_63/Tensordot/concat_1?
dense_63/TensordotReshape#dense_63/Tensordot/MatMul:product:0$dense_63/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_63/Tensordot?
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_63/BiasAdd/ReadVariableOp?
dense_63/BiasAddBiasAdddense_63/Tensordot:output:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_63/BiasAddx
dense_63/ReluReludense_63/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_63/Relu?
dropout_93/IdentityIdentitydense_63/Relu:activations:0*
T0*,
_output_shapes
:??????????2
dropout_93/Identity?
!dense_64/Tensordot/ReadVariableOpReadVariableOp*dense_64_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_64/Tensordot/ReadVariableOp|
dense_64/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_64/Tensordot/axes?
dense_64/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_64/Tensordot/free?
dense_64/Tensordot/ShapeShapedropout_93/Identity:output:0*
T0*
_output_shapes
:2
dense_64/Tensordot/Shape?
 dense_64/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_64/Tensordot/GatherV2/axis?
dense_64/Tensordot/GatherV2GatherV2!dense_64/Tensordot/Shape:output:0 dense_64/Tensordot/free:output:0)dense_64/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_64/Tensordot/GatherV2?
"dense_64/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_64/Tensordot/GatherV2_1/axis?
dense_64/Tensordot/GatherV2_1GatherV2!dense_64/Tensordot/Shape:output:0 dense_64/Tensordot/axes:output:0+dense_64/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_64/Tensordot/GatherV2_1~
dense_64/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_64/Tensordot/Const?
dense_64/Tensordot/ProdProd$dense_64/Tensordot/GatherV2:output:0!dense_64/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_64/Tensordot/Prod?
dense_64/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_64/Tensordot/Const_1?
dense_64/Tensordot/Prod_1Prod&dense_64/Tensordot/GatherV2_1:output:0#dense_64/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_64/Tensordot/Prod_1?
dense_64/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_64/Tensordot/concat/axis?
dense_64/Tensordot/concatConcatV2 dense_64/Tensordot/free:output:0 dense_64/Tensordot/axes:output:0'dense_64/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_64/Tensordot/concat?
dense_64/Tensordot/stackPack dense_64/Tensordot/Prod:output:0"dense_64/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_64/Tensordot/stack?
dense_64/Tensordot/transpose	Transposedropout_93/Identity:output:0"dense_64/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_64/Tensordot/transpose?
dense_64/Tensordot/ReshapeReshape dense_64/Tensordot/transpose:y:0!dense_64/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_64/Tensordot/Reshape?
dense_64/Tensordot/MatMulMatMul#dense_64/Tensordot/Reshape:output:0)dense_64/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_64/Tensordot/MatMul?
dense_64/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_64/Tensordot/Const_2?
 dense_64/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_64/Tensordot/concat_1/axis?
dense_64/Tensordot/concat_1ConcatV2$dense_64/Tensordot/GatherV2:output:0#dense_64/Tensordot/Const_2:output:0)dense_64/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_64/Tensordot/concat_1?
dense_64/TensordotReshape#dense_64/Tensordot/MatMul:product:0$dense_64/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_64/Tensordot?
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_64/BiasAdd/ReadVariableOp?
dense_64/BiasAddBiasAdddense_64/Tensordot:output:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_64/BiasAddx
dense_64/ReluReludense_64/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_64/Relu?
dropout_94/IdentityIdentitydense_64/Relu:activations:0*
T0*,
_output_shapes
:??????????2
dropout_94/Identity?
!dense_65/Tensordot/ReadVariableOpReadVariableOp*dense_65_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_65/Tensordot/ReadVariableOp|
dense_65/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_65/Tensordot/axes?
dense_65/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_65/Tensordot/free?
dense_65/Tensordot/ShapeShapedropout_94/Identity:output:0*
T0*
_output_shapes
:2
dense_65/Tensordot/Shape?
 dense_65/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_65/Tensordot/GatherV2/axis?
dense_65/Tensordot/GatherV2GatherV2!dense_65/Tensordot/Shape:output:0 dense_65/Tensordot/free:output:0)dense_65/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_65/Tensordot/GatherV2?
"dense_65/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_65/Tensordot/GatherV2_1/axis?
dense_65/Tensordot/GatherV2_1GatherV2!dense_65/Tensordot/Shape:output:0 dense_65/Tensordot/axes:output:0+dense_65/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_65/Tensordot/GatherV2_1~
dense_65/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_65/Tensordot/Const?
dense_65/Tensordot/ProdProd$dense_65/Tensordot/GatherV2:output:0!dense_65/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_65/Tensordot/Prod?
dense_65/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_65/Tensordot/Const_1?
dense_65/Tensordot/Prod_1Prod&dense_65/Tensordot/GatherV2_1:output:0#dense_65/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_65/Tensordot/Prod_1?
dense_65/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_65/Tensordot/concat/axis?
dense_65/Tensordot/concatConcatV2 dense_65/Tensordot/free:output:0 dense_65/Tensordot/axes:output:0'dense_65/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_65/Tensordot/concat?
dense_65/Tensordot/stackPack dense_65/Tensordot/Prod:output:0"dense_65/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_65/Tensordot/stack?
dense_65/Tensordot/transpose	Transposedropout_94/Identity:output:0"dense_65/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_65/Tensordot/transpose?
dense_65/Tensordot/ReshapeReshape dense_65/Tensordot/transpose:y:0!dense_65/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_65/Tensordot/Reshape?
dense_65/Tensordot/MatMulMatMul#dense_65/Tensordot/Reshape:output:0)dense_65/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_65/Tensordot/MatMul?
dense_65/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_65/Tensordot/Const_2?
 dense_65/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_65/Tensordot/concat_1/axis?
dense_65/Tensordot/concat_1ConcatV2$dense_65/Tensordot/GatherV2:output:0#dense_65/Tensordot/Const_2:output:0)dense_65/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_65/Tensordot/concat_1?
dense_65/TensordotReshape#dense_65/Tensordot/MatMul:product:0$dense_65/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_65/Tensordot?
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_65/BiasAdd/ReadVariableOp?
dense_65/BiasAddBiasAdddense_65/Tensordot:output:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_65/BiasAddx
IdentityIdentitydense_65/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_63/BiasAdd/ReadVariableOp"^dense_63/Tensordot/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp"^dense_64/Tensordot/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp"^dense_65/Tensordot/ReadVariableOp)^gru_56/gru_cell_56/MatMul/ReadVariableOp+^gru_56/gru_cell_56/MatMul_1/ReadVariableOp"^gru_56/gru_cell_56/ReadVariableOp^gru_56/while)^gru_57/gru_cell_57/MatMul/ReadVariableOp+^gru_57/gru_cell_57/MatMul_1/ReadVariableOp"^gru_57/gru_cell_57/ReadVariableOp^gru_57/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2F
!dense_63/Tensordot/ReadVariableOp!dense_63/Tensordot/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2F
!dense_64/Tensordot/ReadVariableOp!dense_64/Tensordot/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2F
!dense_65/Tensordot/ReadVariableOp!dense_65/Tensordot/ReadVariableOp2T
(gru_56/gru_cell_56/MatMul/ReadVariableOp(gru_56/gru_cell_56/MatMul/ReadVariableOp2X
*gru_56/gru_cell_56/MatMul_1/ReadVariableOp*gru_56/gru_cell_56/MatMul_1/ReadVariableOp2F
!gru_56/gru_cell_56/ReadVariableOp!gru_56/gru_cell_56/ReadVariableOp2
gru_56/whilegru_56/while2T
(gru_57/gru_cell_57/MatMul/ReadVariableOp(gru_57/gru_cell_57/MatMul/ReadVariableOp2X
*gru_57/gru_cell_57/MatMul_1/ReadVariableOp*gru_57/gru_cell_57/MatMul_1/ReadVariableOp2F
!gru_57/gru_cell_57/ReadVariableOp!gru_57/gru_cell_57/ReadVariableOp2
gru_57/whilegru_57/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_3597551
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3597551___redundant_placeholder05
1while_while_cond_3597551___redundant_placeholder15
1while_while_cond_3597551___redundant_placeholder25
1while_while_cond_3597551___redundant_placeholder3
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
f
G__inference_dropout_92_layer_call_and_return_conditional_losses_3601266

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
?
H
,__inference_dropout_93_layer_call_fn_3601311

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
GPU 2J 8? *P
fKRI
G__inference_dropout_93_layer_call_and_return_conditional_losses_35982252
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
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_3601545

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
?
?
*__inference_dense_64_layer_call_fn_3601342

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
GPU 2J 8? *N
fIRG
E__inference_dense_64_layer_call_and_return_conditional_losses_35982582
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
?
?
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_3601506

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
?

?
-__inference_gru_cell_56_layer_call_fn_3601467

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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_35969232
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
?E
?
while_body_3601150
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_57_readvariableop_resource_0:	?F
2while_gru_cell_57_matmul_readvariableop_resource_0:
??H
4while_gru_cell_57_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_57_readvariableop_resource:	?D
0while_gru_cell_57_matmul_readvariableop_resource:
??F
2while_gru_cell_57_matmul_1_readvariableop_resource:
????'while/gru_cell_57/MatMul/ReadVariableOp?)while/gru_cell_57/MatMul_1/ReadVariableOp? while/gru_cell_57/ReadVariableOp?
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
 while/gru_cell_57/ReadVariableOpReadVariableOp+while_gru_cell_57_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_57/ReadVariableOp?
while/gru_cell_57/unstackUnpack(while/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_57/unstack?
'while/gru_cell_57/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_57_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_57/MatMul/ReadVariableOp?
while/gru_cell_57/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul?
while/gru_cell_57/BiasAddBiasAdd"while/gru_cell_57/MatMul:product:0"while/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd?
!while/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_57/split/split_dim?
while/gru_cell_57/splitSplit*while/gru_cell_57/split/split_dim:output:0"while/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split?
)while/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_57_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_57/MatMul_1/ReadVariableOp?
while/gru_cell_57/MatMul_1MatMulwhile_placeholder_21while/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul_1?
while/gru_cell_57/BiasAdd_1BiasAdd$while/gru_cell_57/MatMul_1:product:0"while/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd_1?
while/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_57/Const?
#while/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_57/split_1/split_dim?
while/gru_cell_57/split_1SplitV$while/gru_cell_57/BiasAdd_1:output:0 while/gru_cell_57/Const:output:0,while/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split_1?
while/gru_cell_57/addAddV2 while/gru_cell_57/split:output:0"while/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add?
while/gru_cell_57/SigmoidSigmoidwhile/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid?
while/gru_cell_57/add_1AddV2 while/gru_cell_57/split:output:1"while/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_1?
while/gru_cell_57/Sigmoid_1Sigmoidwhile/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid_1?
while/gru_cell_57/mulMulwhile/gru_cell_57/Sigmoid_1:y:0"while/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul?
while/gru_cell_57/add_2AddV2 while/gru_cell_57/split:output:2while/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_2?
while/gru_cell_57/ReluReluwhile/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Relu?
while/gru_cell_57/mul_1Mulwhile/gru_cell_57/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_1w
while/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_57/sub/x?
while/gru_cell_57/subSub while/gru_cell_57/sub/x:output:0while/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/sub?
while/gru_cell_57/mul_2Mulwhile/gru_cell_57/sub:z:0$while/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_2?
while/gru_cell_57/add_3AddV2while/gru_cell_57/mul_1:z:0while/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_57/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_57/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_57/MatMul/ReadVariableOp*^while/gru_cell_57/MatMul_1/ReadVariableOp!^while/gru_cell_57/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_57_matmul_1_readvariableop_resource4while_gru_cell_57_matmul_1_readvariableop_resource_0"f
0while_gru_cell_57_matmul_readvariableop_resource2while_gru_cell_57_matmul_readvariableop_resource_0"X
)while_gru_cell_57_readvariableop_resource+while_gru_cell_57_readvariableop_resource_0")
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
'while/gru_cell_57/MatMul/ReadVariableOp'while/gru_cell_57/MatMul/ReadVariableOp2V
)while/gru_cell_57/MatMul_1/ReadVariableOp)while/gru_cell_57/MatMul_1/ReadVariableOp2D
 while/gru_cell_57/ReadVariableOp while/gru_cell_57/ReadVariableOp: 
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
while_body_3596793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_56_3596815_0:	?.
while_gru_cell_56_3596817_0:	?/
while_gru_cell_56_3596819_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_56_3596815:	?,
while_gru_cell_56_3596817:	?-
while_gru_cell_56_3596819:
????)while/gru_cell_56/StatefulPartitionedCall?
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
)while/gru_cell_56/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_56_3596815_0while_gru_cell_56_3596817_0while_gru_cell_56_3596819_0*
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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_35967802+
)while/gru_cell_56/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_56/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_56/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_56/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"8
while_gru_cell_56_3596815while_gru_cell_56_3596815_0"8
while_gru_cell_56_3596817while_gru_cell_56_3596817_0"8
while_gru_cell_56_3596819while_gru_cell_56_3596819_0")
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
)while/gru_cell_56/StatefulPartitionedCall)while/gru_cell_56/StatefulPartitionedCall: 
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
(__inference_gru_57_layer_call_fn_3600627

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
GPU 2J 8? *L
fGRE
C__inference_gru_57_layer_call_and_return_conditional_losses_35986002
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
?*
?
J__inference_sequential_28_layer_call_and_return_conditional_losses_3598968
gru_56_input!
gru_56_3598934:	?!
gru_56_3598936:	?"
gru_56_3598938:
??!
gru_57_3598942:	?"
gru_57_3598944:
??"
gru_57_3598946:
??$
dense_63_3598950:
??
dense_63_3598952:	?$
dense_64_3598956:
??
dense_64_3598958:	?#
dense_65_3598962:	?
dense_65_3598964:
identity?? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall?gru_56/StatefulPartitionedCall?gru_57/StatefulPartitionedCall?
gru_56/StatefulPartitionedCallStatefulPartitionedCallgru_56_inputgru_56_3598934gru_56_3598936gru_56_3598938*
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
GPU 2J 8? *L
fGRE
C__inference_gru_56_layer_call_and_return_conditional_losses_35980012 
gru_56/StatefulPartitionedCall?
dropout_91/PartitionedCallPartitionedCall'gru_56/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_35980142
dropout_91/PartitionedCall?
gru_57/StatefulPartitionedCallStatefulPartitionedCall#dropout_91/PartitionedCall:output:0gru_57_3598942gru_57_3598944gru_57_3598946*
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
GPU 2J 8? *L
fGRE
C__inference_gru_57_layer_call_and_return_conditional_losses_35981682 
gru_57/StatefulPartitionedCall?
dropout_92/PartitionedCallPartitionedCall'gru_57/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_35981812
dropout_92/PartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall#dropout_92/PartitionedCall:output:0dense_63_3598950dense_63_3598952*
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
GPU 2J 8? *N
fIRG
E__inference_dense_63_layer_call_and_return_conditional_losses_35982142"
 dense_63/StatefulPartitionedCall?
dropout_93/PartitionedCallPartitionedCall)dense_63/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_93_layer_call_and_return_conditional_losses_35982252
dropout_93/PartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall#dropout_93/PartitionedCall:output:0dense_64_3598956dense_64_3598958*
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
GPU 2J 8? *N
fIRG
E__inference_dense_64_layer_call_and_return_conditional_losses_35982582"
 dense_64/StatefulPartitionedCall?
dropout_94/PartitionedCallPartitionedCall)dense_64/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_94_layer_call_and_return_conditional_losses_35982692
dropout_94/PartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall#dropout_94/PartitionedCall:output:0dense_65_3598962dense_65_3598964*
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
GPU 2J 8? *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_35983012"
 dense_65/StatefulPartitionedCall?
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall^gru_56/StatefulPartitionedCall^gru_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2@
gru_56/StatefulPartitionedCallgru_56/StatefulPartitionedCall2@
gru_57/StatefulPartitionedCallgru_57/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_56_input
?"
?
while_body_3596986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_56_3597008_0:	?.
while_gru_cell_56_3597010_0:	?/
while_gru_cell_56_3597012_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_56_3597008:	?,
while_gru_cell_56_3597010:	?-
while_gru_cell_56_3597012:
????)while/gru_cell_56/StatefulPartitionedCall?
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
)while/gru_cell_56/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_56_3597008_0while_gru_cell_56_3597010_0while_gru_cell_56_3597012_0*
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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_35969232+
)while/gru_cell_56/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_56/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_56/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_56/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"8
while_gru_cell_56_3597008while_gru_cell_56_3597008_0"8
while_gru_cell_56_3597010while_gru_cell_56_3597010_0"8
while_gru_cell_56_3597012while_gru_cell_56_3597012_0")
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
)while/gru_cell_56/StatefulPartitionedCall)while/gru_cell_56/StatefulPartitionedCall: 
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
-__inference_gru_cell_56_layer_call_fn_3601453

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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_35967802
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
?X
?
C__inference_gru_56_layer_call_and_return_conditional_losses_3600403

inputs6
#gru_cell_56_readvariableop_resource:	?=
*gru_cell_56_matmul_readvariableop_resource:	?@
,gru_cell_56_matmul_1_readvariableop_resource:
??
identity??!gru_cell_56/MatMul/ReadVariableOp?#gru_cell_56/MatMul_1/ReadVariableOp?gru_cell_56/ReadVariableOp?whileD
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
gru_cell_56/ReadVariableOpReadVariableOp#gru_cell_56_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_56/ReadVariableOp?
gru_cell_56/unstackUnpack"gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_56/unstack?
!gru_cell_56/MatMul/ReadVariableOpReadVariableOp*gru_cell_56_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_56/MatMul/ReadVariableOp?
gru_cell_56/MatMulMatMulstrided_slice_2:output:0)gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul?
gru_cell_56/BiasAddBiasAddgru_cell_56/MatMul:product:0gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd?
gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split/split_dim?
gru_cell_56/splitSplit$gru_cell_56/split/split_dim:output:0gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split?
#gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_56_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_56/MatMul_1/ReadVariableOp?
gru_cell_56/MatMul_1MatMulzeros:output:0+gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul_1?
gru_cell_56/BiasAdd_1BiasAddgru_cell_56/MatMul_1:product:0gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd_1{
gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_56/Const?
gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split_1/split_dim?
gru_cell_56/split_1SplitVgru_cell_56/BiasAdd_1:output:0gru_cell_56/Const:output:0&gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split_1?
gru_cell_56/addAddV2gru_cell_56/split:output:0gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add}
gru_cell_56/SigmoidSigmoidgru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid?
gru_cell_56/add_1AddV2gru_cell_56/split:output:1gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_1?
gru_cell_56/Sigmoid_1Sigmoidgru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid_1?
gru_cell_56/mulMulgru_cell_56/Sigmoid_1:y:0gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul?
gru_cell_56/add_2AddV2gru_cell_56/split:output:2gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_2v
gru_cell_56/ReluRelugru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Relu?
gru_cell_56/mul_1Mulgru_cell_56/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_1k
gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_56/sub/x?
gru_cell_56/subSubgru_cell_56/sub/x:output:0gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/sub?
gru_cell_56/mul_2Mulgru_cell_56/sub:z:0gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_2?
gru_cell_56/add_3AddV2gru_cell_56/mul_1:z:0gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_56_readvariableop_resource*gru_cell_56_matmul_readvariableop_resource,gru_cell_56_matmul_1_readvariableop_resource*
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
bodyR
while_body_3600314*
condR
while_cond_3600313*9
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
NoOpNoOp"^gru_cell_56/MatMul/ReadVariableOp$^gru_cell_56/MatMul_1/ReadVariableOp^gru_cell_56/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_56/MatMul/ReadVariableOp!gru_cell_56/MatMul/ReadVariableOp2J
#gru_cell_56/MatMul_1/ReadVariableOp#gru_cell_56/MatMul_1/ReadVariableOp28
gru_cell_56/ReadVariableOpgru_cell_56/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
while_body_3598079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_57_readvariableop_resource_0:	?F
2while_gru_cell_57_matmul_readvariableop_resource_0:
??H
4while_gru_cell_57_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_57_readvariableop_resource:	?D
0while_gru_cell_57_matmul_readvariableop_resource:
??F
2while_gru_cell_57_matmul_1_readvariableop_resource:
????'while/gru_cell_57/MatMul/ReadVariableOp?)while/gru_cell_57/MatMul_1/ReadVariableOp? while/gru_cell_57/ReadVariableOp?
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
 while/gru_cell_57/ReadVariableOpReadVariableOp+while_gru_cell_57_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_57/ReadVariableOp?
while/gru_cell_57/unstackUnpack(while/gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_57/unstack?
'while/gru_cell_57/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_57_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_57/MatMul/ReadVariableOp?
while/gru_cell_57/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul?
while/gru_cell_57/BiasAddBiasAdd"while/gru_cell_57/MatMul:product:0"while/gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd?
!while/gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_57/split/split_dim?
while/gru_cell_57/splitSplit*while/gru_cell_57/split/split_dim:output:0"while/gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split?
)while/gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_57_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_57/MatMul_1/ReadVariableOp?
while/gru_cell_57/MatMul_1MatMulwhile_placeholder_21while/gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/MatMul_1?
while/gru_cell_57/BiasAdd_1BiasAdd$while/gru_cell_57/MatMul_1:product:0"while/gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/BiasAdd_1?
while/gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_57/Const?
#while/gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_57/split_1/split_dim?
while/gru_cell_57/split_1SplitV$while/gru_cell_57/BiasAdd_1:output:0 while/gru_cell_57/Const:output:0,while/gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_57/split_1?
while/gru_cell_57/addAddV2 while/gru_cell_57/split:output:0"while/gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add?
while/gru_cell_57/SigmoidSigmoidwhile/gru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid?
while/gru_cell_57/add_1AddV2 while/gru_cell_57/split:output:1"while/gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_1?
while/gru_cell_57/Sigmoid_1Sigmoidwhile/gru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Sigmoid_1?
while/gru_cell_57/mulMulwhile/gru_cell_57/Sigmoid_1:y:0"while/gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul?
while/gru_cell_57/add_2AddV2 while/gru_cell_57/split:output:2while/gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_2?
while/gru_cell_57/ReluReluwhile/gru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/Relu?
while/gru_cell_57/mul_1Mulwhile/gru_cell_57/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_1w
while/gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_57/sub/x?
while/gru_cell_57/subSub while/gru_cell_57/sub/x:output:0while/gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/sub?
while/gru_cell_57/mul_2Mulwhile/gru_cell_57/sub:z:0$while/gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/mul_2?
while/gru_cell_57/add_3AddV2while/gru_cell_57/mul_1:z:0while/gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_57/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_57/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_57/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_57/MatMul/ReadVariableOp*^while/gru_cell_57/MatMul_1/ReadVariableOp!^while/gru_cell_57/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_57_matmul_1_readvariableop_resource4while_gru_cell_57_matmul_1_readvariableop_resource_0"f
0while_gru_cell_57_matmul_readvariableop_resource2while_gru_cell_57_matmul_readvariableop_resource_0"X
)while_gru_cell_57_readvariableop_resource+while_gru_cell_57_readvariableop_resource_0")
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
'while/gru_cell_57/MatMul/ReadVariableOp'while/gru_cell_57/MatMul/ReadVariableOp2V
)while/gru_cell_57/MatMul_1/ReadVariableOp)while/gru_cell_57/MatMul_1/ReadVariableOp2D
 while/gru_cell_57/ReadVariableOp while/gru_cell_57/ReadVariableOp: 
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
?
while_cond_3597358
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3597358___redundant_placeholder05
1while_while_cond_3597358___redundant_placeholder15
1while_while_cond_3597358___redundant_placeholder25
1while_while_cond_3597358___redundant_placeholder3
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
-__inference_gru_cell_57_layer_call_fn_3601559

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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_35973462
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
?
f
G__inference_dropout_93_layer_call_and_return_conditional_losses_3601333

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
?
while_cond_3598708
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3598708___redundant_placeholder05
1while_while_cond_3598708___redundant_placeholder15
1while_while_cond_3598708___redundant_placeholder25
1while_while_cond_3598708___redundant_placeholder3
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
(__inference_gru_56_layer_call_fn_3599922
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
GPU 2J 8? *L
fGRE
C__inference_gru_56_layer_call_and_return_conditional_losses_35970502
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
f
G__inference_dropout_94_layer_call_and_return_conditional_losses_3598365

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
?0
?
J__inference_sequential_28_layer_call_and_return_conditional_losses_3598875

inputs!
gru_56_3598841:	?!
gru_56_3598843:	?"
gru_56_3598845:
??!
gru_57_3598849:	?"
gru_57_3598851:
??"
gru_57_3598853:
??$
dense_63_3598857:
??
dense_63_3598859:	?$
dense_64_3598863:
??
dense_64_3598865:	?#
dense_65_3598869:	?
dense_65_3598871:
identity?? dense_63/StatefulPartitionedCall? dense_64/StatefulPartitionedCall? dense_65/StatefulPartitionedCall?"dropout_91/StatefulPartitionedCall?"dropout_92/StatefulPartitionedCall?"dropout_93/StatefulPartitionedCall?"dropout_94/StatefulPartitionedCall?gru_56/StatefulPartitionedCall?gru_57/StatefulPartitionedCall?
gru_56/StatefulPartitionedCallStatefulPartitionedCallinputsgru_56_3598841gru_56_3598843gru_56_3598845*
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
GPU 2J 8? *L
fGRE
C__inference_gru_56_layer_call_and_return_conditional_losses_35987982 
gru_56/StatefulPartitionedCall?
"dropout_91/StatefulPartitionedCallStatefulPartitionedCall'gru_56/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_35986292$
"dropout_91/StatefulPartitionedCall?
gru_57/StatefulPartitionedCallStatefulPartitionedCall+dropout_91/StatefulPartitionedCall:output:0gru_57_3598849gru_57_3598851gru_57_3598853*
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
GPU 2J 8? *L
fGRE
C__inference_gru_57_layer_call_and_return_conditional_losses_35986002 
gru_57/StatefulPartitionedCall?
"dropout_92/StatefulPartitionedCallStatefulPartitionedCall'gru_57/StatefulPartitionedCall:output:0#^dropout_91/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_35984312$
"dropout_92/StatefulPartitionedCall?
 dense_63/StatefulPartitionedCallStatefulPartitionedCall+dropout_92/StatefulPartitionedCall:output:0dense_63_3598857dense_63_3598859*
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
GPU 2J 8? *N
fIRG
E__inference_dense_63_layer_call_and_return_conditional_losses_35982142"
 dense_63/StatefulPartitionedCall?
"dropout_93/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0#^dropout_92/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_93_layer_call_and_return_conditional_losses_35983982$
"dropout_93/StatefulPartitionedCall?
 dense_64/StatefulPartitionedCallStatefulPartitionedCall+dropout_93/StatefulPartitionedCall:output:0dense_64_3598863dense_64_3598865*
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
GPU 2J 8? *N
fIRG
E__inference_dense_64_layer_call_and_return_conditional_losses_35982582"
 dense_64/StatefulPartitionedCall?
"dropout_94/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0#^dropout_93/StatefulPartitionedCall*
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
GPU 2J 8? *P
fKRI
G__inference_dropout_94_layer_call_and_return_conditional_losses_35983652$
"dropout_94/StatefulPartitionedCall?
 dense_65/StatefulPartitionedCallStatefulPartitionedCall+dropout_94/StatefulPartitionedCall:output:0dense_65_3598869dense_65_3598871*
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
GPU 2J 8? *N
fIRG
E__inference_dense_65_layer_call_and_return_conditional_losses_35983012"
 dense_65/StatefulPartitionedCall?
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall#^dropout_91/StatefulPartitionedCall#^dropout_92/StatefulPartitionedCall#^dropout_93/StatefulPartitionedCall#^dropout_94/StatefulPartitionedCall^gru_56/StatefulPartitionedCall^gru_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2H
"dropout_91/StatefulPartitionedCall"dropout_91/StatefulPartitionedCall2H
"dropout_92/StatefulPartitionedCall"dropout_92/StatefulPartitionedCall2H
"dropout_93/StatefulPartitionedCall"dropout_93/StatefulPartitionedCall2H
"dropout_94/StatefulPartitionedCall"dropout_94/StatefulPartitionedCall2@
gru_56/StatefulPartitionedCallgru_56/StatefulPartitionedCall2@
gru_57/StatefulPartitionedCallgru_57/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
C__inference_gru_57_layer_call_and_return_conditional_losses_3597423

inputs&
gru_cell_57_3597347:	?'
gru_cell_57_3597349:
??'
gru_cell_57_3597351:
??
identity??#gru_cell_57/StatefulPartitionedCall?whileD
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
#gru_cell_57/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_57_3597347gru_cell_57_3597349gru_cell_57_3597351*
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
GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_35973462%
#gru_cell_57/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_57_3597347gru_cell_57_3597349gru_cell_57_3597351*
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
bodyR
while_body_3597359*
condR
while_cond_3597358*9
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
NoOpNoOp$^gru_cell_57/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#gru_cell_57/StatefulPartitionedCall#gru_cell_57/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_28_layer_call_fn_3598931
gru_56_input
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
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_56_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_28_layer_call_and_return_conditional_losses_35988752
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
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_56_input
?
?
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_3596780

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
?X
?
C__inference_gru_57_layer_call_and_return_conditional_losses_3598600

inputs6
#gru_cell_57_readvariableop_resource:	?>
*gru_cell_57_matmul_readvariableop_resource:
??@
,gru_cell_57_matmul_1_readvariableop_resource:
??
identity??!gru_cell_57/MatMul/ReadVariableOp?#gru_cell_57/MatMul_1/ReadVariableOp?gru_cell_57/ReadVariableOp?whileD
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
gru_cell_57/ReadVariableOpReadVariableOp#gru_cell_57_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_57/ReadVariableOp?
gru_cell_57/unstackUnpack"gru_cell_57/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_57/unstack?
!gru_cell_57/MatMul/ReadVariableOpReadVariableOp*gru_cell_57_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_57/MatMul/ReadVariableOp?
gru_cell_57/MatMulMatMulstrided_slice_2:output:0)gru_cell_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul?
gru_cell_57/BiasAddBiasAddgru_cell_57/MatMul:product:0gru_cell_57/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd?
gru_cell_57/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split/split_dim?
gru_cell_57/splitSplit$gru_cell_57/split/split_dim:output:0gru_cell_57/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split?
#gru_cell_57/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_57_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_57/MatMul_1/ReadVariableOp?
gru_cell_57/MatMul_1MatMulzeros:output:0+gru_cell_57/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/MatMul_1?
gru_cell_57/BiasAdd_1BiasAddgru_cell_57/MatMul_1:product:0gru_cell_57/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/BiasAdd_1{
gru_cell_57/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_57/Const?
gru_cell_57/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_57/split_1/split_dim?
gru_cell_57/split_1SplitVgru_cell_57/BiasAdd_1:output:0gru_cell_57/Const:output:0&gru_cell_57/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_57/split_1?
gru_cell_57/addAddV2gru_cell_57/split:output:0gru_cell_57/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add}
gru_cell_57/SigmoidSigmoidgru_cell_57/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid?
gru_cell_57/add_1AddV2gru_cell_57/split:output:1gru_cell_57/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_1?
gru_cell_57/Sigmoid_1Sigmoidgru_cell_57/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Sigmoid_1?
gru_cell_57/mulMulgru_cell_57/Sigmoid_1:y:0gru_cell_57/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul?
gru_cell_57/add_2AddV2gru_cell_57/split:output:2gru_cell_57/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_2v
gru_cell_57/ReluRelugru_cell_57/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/Relu?
gru_cell_57/mul_1Mulgru_cell_57/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_1k
gru_cell_57/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_57/sub/x?
gru_cell_57/subSubgru_cell_57/sub/x:output:0gru_cell_57/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/sub?
gru_cell_57/mul_2Mulgru_cell_57/sub:z:0gru_cell_57/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/mul_2?
gru_cell_57/add_3AddV2gru_cell_57/mul_1:z:0gru_cell_57/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_57/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_57_readvariableop_resource*gru_cell_57_matmul_readvariableop_resource,gru_cell_57_matmul_1_readvariableop_resource*
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
bodyR
while_body_3598511*
condR
while_cond_3598510*9
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
NoOpNoOp"^gru_cell_57/MatMul/ReadVariableOp$^gru_cell_57/MatMul_1/ReadVariableOp^gru_cell_57/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_57/MatMul/ReadVariableOp!gru_cell_57/MatMul/ReadVariableOp2J
#gru_cell_57/MatMul_1/ReadVariableOp#gru_cell_57/MatMul_1/ReadVariableOp28
gru_cell_57/ReadVariableOpgru_cell_57/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_3600690
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3600690___redundant_placeholder05
1while_while_cond_3600690___redundant_placeholder15
1while_while_cond_3600690___redundant_placeholder25
1while_while_cond_3600690___redundant_placeholder3
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
?!
?
E__inference_dense_64_layer_call_and_return_conditional_losses_3598258

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
?X
?
C__inference_gru_56_layer_call_and_return_conditional_losses_3598798

inputs6
#gru_cell_56_readvariableop_resource:	?=
*gru_cell_56_matmul_readvariableop_resource:	?@
,gru_cell_56_matmul_1_readvariableop_resource:
??
identity??!gru_cell_56/MatMul/ReadVariableOp?#gru_cell_56/MatMul_1/ReadVariableOp?gru_cell_56/ReadVariableOp?whileD
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
gru_cell_56/ReadVariableOpReadVariableOp#gru_cell_56_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_56/ReadVariableOp?
gru_cell_56/unstackUnpack"gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_56/unstack?
!gru_cell_56/MatMul/ReadVariableOpReadVariableOp*gru_cell_56_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_56/MatMul/ReadVariableOp?
gru_cell_56/MatMulMatMulstrided_slice_2:output:0)gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul?
gru_cell_56/BiasAddBiasAddgru_cell_56/MatMul:product:0gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd?
gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split/split_dim?
gru_cell_56/splitSplit$gru_cell_56/split/split_dim:output:0gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split?
#gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_56_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_56/MatMul_1/ReadVariableOp?
gru_cell_56/MatMul_1MatMulzeros:output:0+gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/MatMul_1?
gru_cell_56/BiasAdd_1BiasAddgru_cell_56/MatMul_1:product:0gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/BiasAdd_1{
gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_cell_56/Const?
gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_56/split_1/split_dim?
gru_cell_56/split_1SplitVgru_cell_56/BiasAdd_1:output:0gru_cell_56/Const:output:0&gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_56/split_1?
gru_cell_56/addAddV2gru_cell_56/split:output:0gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add}
gru_cell_56/SigmoidSigmoidgru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid?
gru_cell_56/add_1AddV2gru_cell_56/split:output:1gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_1?
gru_cell_56/Sigmoid_1Sigmoidgru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Sigmoid_1?
gru_cell_56/mulMulgru_cell_56/Sigmoid_1:y:0gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul?
gru_cell_56/add_2AddV2gru_cell_56/split:output:2gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_2v
gru_cell_56/ReluRelugru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/Relu?
gru_cell_56/mul_1Mulgru_cell_56/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_1k
gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_56/sub/x?
gru_cell_56/subSubgru_cell_56/sub/x:output:0gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/sub?
gru_cell_56/mul_2Mulgru_cell_56/sub:z:0gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/mul_2?
gru_cell_56/add_3AddV2gru_cell_56/mul_1:z:0gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_56/add_3?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_56_readvariableop_resource*gru_cell_56_matmul_readvariableop_resource,gru_cell_56_matmul_1_readvariableop_resource*
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
bodyR
while_body_3598709*
condR
while_cond_3598708*9
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
NoOpNoOp"^gru_cell_56/MatMul/ReadVariableOp$^gru_cell_56/MatMul_1/ReadVariableOp^gru_cell_56/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_56/MatMul/ReadVariableOp!gru_cell_56/MatMul/ReadVariableOp2J
#gru_cell_56/MatMul_1/ReadVariableOp#gru_cell_56/MatMul_1/ReadVariableOp28
gru_cell_56/ReadVariableOpgru_cell_56/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
while_body_3597912
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_56_readvariableop_resource_0:	?E
2while_gru_cell_56_matmul_readvariableop_resource_0:	?H
4while_gru_cell_56_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_56_readvariableop_resource:	?C
0while_gru_cell_56_matmul_readvariableop_resource:	?F
2while_gru_cell_56_matmul_1_readvariableop_resource:
????'while/gru_cell_56/MatMul/ReadVariableOp?)while/gru_cell_56/MatMul_1/ReadVariableOp? while/gru_cell_56/ReadVariableOp?
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
 while/gru_cell_56/ReadVariableOpReadVariableOp+while_gru_cell_56_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_56/ReadVariableOp?
while/gru_cell_56/unstackUnpack(while/gru_cell_56/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_56/unstack?
'while/gru_cell_56/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_56_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_56/MatMul/ReadVariableOp?
while/gru_cell_56/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul?
while/gru_cell_56/BiasAddBiasAdd"while/gru_cell_56/MatMul:product:0"while/gru_cell_56/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd?
!while/gru_cell_56/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_56/split/split_dim?
while/gru_cell_56/splitSplit*while/gru_cell_56/split/split_dim:output:0"while/gru_cell_56/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split?
)while/gru_cell_56/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_56_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_56/MatMul_1/ReadVariableOp?
while/gru_cell_56/MatMul_1MatMulwhile_placeholder_21while/gru_cell_56/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/MatMul_1?
while/gru_cell_56/BiasAdd_1BiasAdd$while/gru_cell_56/MatMul_1:product:0"while/gru_cell_56/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/BiasAdd_1?
while/gru_cell_56/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ????2
while/gru_cell_56/Const?
#while/gru_cell_56/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_56/split_1/split_dim?
while/gru_cell_56/split_1SplitV$while/gru_cell_56/BiasAdd_1:output:0 while/gru_cell_56/Const:output:0,while/gru_cell_56/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_56/split_1?
while/gru_cell_56/addAddV2 while/gru_cell_56/split:output:0"while/gru_cell_56/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add?
while/gru_cell_56/SigmoidSigmoidwhile/gru_cell_56/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid?
while/gru_cell_56/add_1AddV2 while/gru_cell_56/split:output:1"while/gru_cell_56/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_1?
while/gru_cell_56/Sigmoid_1Sigmoidwhile/gru_cell_56/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Sigmoid_1?
while/gru_cell_56/mulMulwhile/gru_cell_56/Sigmoid_1:y:0"while/gru_cell_56/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul?
while/gru_cell_56/add_2AddV2 while/gru_cell_56/split:output:2while/gru_cell_56/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_2?
while/gru_cell_56/ReluReluwhile/gru_cell_56/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/Relu?
while/gru_cell_56/mul_1Mulwhile/gru_cell_56/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_1w
while/gru_cell_56/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_56/sub/x?
while/gru_cell_56/subSub while/gru_cell_56/sub/x:output:0while/gru_cell_56/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/sub?
while/gru_cell_56/mul_2Mulwhile/gru_cell_56/sub:z:0$while/gru_cell_56/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/mul_2?
while/gru_cell_56/add_3AddV2while/gru_cell_56/mul_1:z:0while/gru_cell_56/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_56/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_56/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_56/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_56/MatMul/ReadVariableOp*^while/gru_cell_56/MatMul_1/ReadVariableOp!^while/gru_cell_56/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_56_matmul_1_readvariableop_resource4while_gru_cell_56_matmul_1_readvariableop_resource_0"f
0while_gru_cell_56_matmul_readvariableop_resource2while_gru_cell_56_matmul_readvariableop_resource_0"X
)while_gru_cell_56_readvariableop_resource+while_gru_cell_56_readvariableop_resource_0")
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
'while/gru_cell_56/MatMul/ReadVariableOp'while/gru_cell_56/MatMul/ReadVariableOp2V
)while/gru_cell_56/MatMul_1/ReadVariableOp)while/gru_cell_56/MatMul_1/ReadVariableOp2D
 while/gru_cell_56/ReadVariableOp while/gru_cell_56/ReadVariableOp: 
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
?
gru_57_while_cond_3599706*
&gru_57_while_gru_57_while_loop_counter0
,gru_57_while_gru_57_while_maximum_iterations
gru_57_while_placeholder
gru_57_while_placeholder_1
gru_57_while_placeholder_2,
(gru_57_while_less_gru_57_strided_slice_1C
?gru_57_while_gru_57_while_cond_3599706___redundant_placeholder0C
?gru_57_while_gru_57_while_cond_3599706___redundant_placeholder1C
?gru_57_while_gru_57_while_cond_3599706___redundant_placeholder2C
?gru_57_while_gru_57_while_cond_3599706___redundant_placeholder3
gru_57_while_identity
?
gru_57/while/LessLessgru_57_while_placeholder(gru_57_while_less_gru_57_strided_slice_1*
T0*
_output_shapes
: 2
gru_57/while/Lessr
gru_57/while/IdentityIdentitygru_57/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_57/while/Identity"7
gru_57_while_identitygru_57/while/Identity:output:0*(
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
(__inference_gru_56_layer_call_fn_3599944

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
GPU 2J 8? *L
fGRE
C__inference_gru_56_layer_call_and_return_conditional_losses_35987982
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
?
?
while_cond_3601149
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3601149___redundant_placeholder05
1while_while_cond_3601149___redundant_placeholder15
1while_while_cond_3601149___redundant_placeholder25
1while_while_cond_3601149___redundant_placeholder3
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
?
while_cond_3598078
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3598078___redundant_placeholder05
1while_while_cond_3598078___redundant_placeholder15
1while_while_cond_3598078___redundant_placeholder25
1while_while_cond_3598078___redundant_placeholder3
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
?
while_cond_3596792
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3596792___redundant_placeholder05
1while_while_cond_3596792___redundant_placeholder15
1while_while_cond_3596792___redundant_placeholder25
1while_while_cond_3596792___redundant_placeholder3
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
(__inference_gru_56_layer_call_fn_3599911
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
GPU 2J 8? *L
fGRE
C__inference_gru_56_layer_call_and_return_conditional_losses_35968572
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
?
?
while_cond_3600996
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_3600996___redundant_placeholder05
1while_while_cond_3600996___redundant_placeholder15
1while_while_cond_3600996___redundant_placeholder25
1while_while_cond_3600996___redundant_placeholder3
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
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
gru_56_input9
serving_default_gru_56_input:0?????????@
dense_654
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
layer-7
	layer_with_weights-4
	layer-8

	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
 	variables
!regularization_losses
"trainable_variables
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
*	variables
+regularization_losses
,trainable_variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
4	variables
5regularization_losses
6trainable_variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate$m?%m?.m?/m?8m?9m?Cm?Dm?Em?Fm?Gm?Hm?$v?%v?.v?/v?8v?9v?Cv?Dv?Ev?Fv?Gv?Hv?"
	optimizer
v
C0
D1
E2
F3
G4
H5
$6
%7
.8
/9
810
911"
trackable_list_wrapper
v
C0
D1
E2
F3
G4
H5
$6
%7
.8
/9
810
911"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables

Ilayers
Jlayer_regularization_losses
Kmetrics
trainable_variables
Lnon_trainable_variables
Mlayer_metrics
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

Ckernel
Drecurrent_kernel
Ebias
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables

Rlayers
Slayer_regularization_losses

Tstates
Umetrics
trainable_variables
Vnon_trainable_variables
Wlayer_metrics
regularization_losses
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
	variables
Xlayer_regularization_losses

Ylayers
regularization_losses
Zmetrics
trainable_variables
[layer_metrics
\non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Fkernel
Grecurrent_kernel
Hbias
]	variables
^regularization_losses
_trainable_variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables

alayers
blayer_regularization_losses

cstates
dmetrics
trainable_variables
enon_trainable_variables
flayer_metrics
regularization_losses
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
 	variables
glayer_regularization_losses

hlayers
!regularization_losses
imetrics
"trainable_variables
jlayer_metrics
knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_63/kernel
:?2dense_63/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
&	variables
llayer_regularization_losses

mlayers
'regularization_losses
nmetrics
(trainable_variables
olayer_metrics
pnon_trainable_variables
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
*	variables
qlayer_regularization_losses

rlayers
+regularization_losses
smetrics
,trainable_variables
tlayer_metrics
unon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_64/kernel
:?2dense_64/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
0	variables
vlayer_regularization_losses

wlayers
1regularization_losses
xmetrics
2trainable_variables
ylayer_metrics
znon_trainable_variables
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
4	variables
{layer_regularization_losses

|layers
5regularization_losses
}metrics
6trainable_variables
~layer_metrics
non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_65/kernel
:2dense_65/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
:	variables
 ?layer_regularization_losses
?layers
;regularization_losses
?metrics
<trainable_variables
?layer_metrics
?non_trainable_variables
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
,:*	?2gru_56/gru_cell_56/kernel
7:5
??2#gru_56/gru_cell_56/recurrent_kernel
*:(	?2gru_56/gru_cell_56/bias
-:+
??2gru_57/gru_cell_57/kernel
7:5
??2#gru_57/gru_cell_57/recurrent_kernel
*:(	?2gru_57/gru_cell_57/bias
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
?
N	variables
 ?layer_regularization_losses
?layers
Oregularization_losses
?metrics
Ptrainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
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
trackable_dict_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
?
]	variables
 ?layer_regularization_losses
?layers
^regularization_losses
?metrics
_trainable_variables
?layer_metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
(:&
??2Adam/dense_63/kernel/m
!:?2Adam/dense_63/bias/m
(:&
??2Adam/dense_64/kernel/m
!:?2Adam/dense_64/bias/m
':%	?2Adam/dense_65/kernel/m
 :2Adam/dense_65/bias/m
1:/	?2 Adam/gru_56/gru_cell_56/kernel/m
<::
??2*Adam/gru_56/gru_cell_56/recurrent_kernel/m
/:-	?2Adam/gru_56/gru_cell_56/bias/m
2:0
??2 Adam/gru_57/gru_cell_57/kernel/m
<::
??2*Adam/gru_57/gru_cell_57/recurrent_kernel/m
/:-	?2Adam/gru_57/gru_cell_57/bias/m
(:&
??2Adam/dense_63/kernel/v
!:?2Adam/dense_63/bias/v
(:&
??2Adam/dense_64/kernel/v
!:?2Adam/dense_64/bias/v
':%	?2Adam/dense_65/kernel/v
 :2Adam/dense_65/bias/v
1:/	?2 Adam/gru_56/gru_cell_56/kernel/v
<::
??2*Adam/gru_56/gru_cell_56/recurrent_kernel/v
/:-	?2Adam/gru_56/gru_cell_56/bias/v
2:0
??2 Adam/gru_57/gru_cell_57/kernel/v
<::
??2*Adam/gru_57/gru_cell_57/recurrent_kernel/v
/:-	?2Adam/gru_57/gru_cell_57/bias/v
?B?
"__inference__wrapped_model_3596710gru_56_input"?
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
?2?
/__inference_sequential_28_layer_call_fn_3598335
/__inference_sequential_28_layer_call_fn_3599071
/__inference_sequential_28_layer_call_fn_3599100
/__inference_sequential_28_layer_call_fn_3598931?
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
J__inference_sequential_28_layer_call_and_return_conditional_losses_3599486
J__inference_sequential_28_layer_call_and_return_conditional_losses_3599900
J__inference_sequential_28_layer_call_and_return_conditional_losses_3598968
J__inference_sequential_28_layer_call_and_return_conditional_losses_3599005?
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
(__inference_gru_56_layer_call_fn_3599911
(__inference_gru_56_layer_call_fn_3599922
(__inference_gru_56_layer_call_fn_3599933
(__inference_gru_56_layer_call_fn_3599944?
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
?2?
C__inference_gru_56_layer_call_and_return_conditional_losses_3600097
C__inference_gru_56_layer_call_and_return_conditional_losses_3600250
C__inference_gru_56_layer_call_and_return_conditional_losses_3600403
C__inference_gru_56_layer_call_and_return_conditional_losses_3600556?
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
,__inference_dropout_91_layer_call_fn_3600561
,__inference_dropout_91_layer_call_fn_3600566?
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
G__inference_dropout_91_layer_call_and_return_conditional_losses_3600571
G__inference_dropout_91_layer_call_and_return_conditional_losses_3600583?
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
(__inference_gru_57_layer_call_fn_3600594
(__inference_gru_57_layer_call_fn_3600605
(__inference_gru_57_layer_call_fn_3600616
(__inference_gru_57_layer_call_fn_3600627?
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
?2?
C__inference_gru_57_layer_call_and_return_conditional_losses_3600780
C__inference_gru_57_layer_call_and_return_conditional_losses_3600933
C__inference_gru_57_layer_call_and_return_conditional_losses_3601086
C__inference_gru_57_layer_call_and_return_conditional_losses_3601239?
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
,__inference_dropout_92_layer_call_fn_3601244
,__inference_dropout_92_layer_call_fn_3601249?
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
G__inference_dropout_92_layer_call_and_return_conditional_losses_3601254
G__inference_dropout_92_layer_call_and_return_conditional_losses_3601266?
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
*__inference_dense_63_layer_call_fn_3601275?
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
E__inference_dense_63_layer_call_and_return_conditional_losses_3601306?
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
,__inference_dropout_93_layer_call_fn_3601311
,__inference_dropout_93_layer_call_fn_3601316?
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
G__inference_dropout_93_layer_call_and_return_conditional_losses_3601321
G__inference_dropout_93_layer_call_and_return_conditional_losses_3601333?
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
*__inference_dense_64_layer_call_fn_3601342?
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
E__inference_dense_64_layer_call_and_return_conditional_losses_3601373?
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
,__inference_dropout_94_layer_call_fn_3601378
,__inference_dropout_94_layer_call_fn_3601383?
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
G__inference_dropout_94_layer_call_and_return_conditional_losses_3601388
G__inference_dropout_94_layer_call_and_return_conditional_losses_3601400?
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
*__inference_dense_65_layer_call_fn_3601409?
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
E__inference_dense_65_layer_call_and_return_conditional_losses_3601439?
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
%__inference_signature_wrapper_3599042gru_56_input"?
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
-__inference_gru_cell_56_layer_call_fn_3601453
-__inference_gru_cell_56_layer_call_fn_3601467?
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
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_3601506
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_3601545?
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
-__inference_gru_cell_57_layer_call_fn_3601559
-__inference_gru_cell_57_layer_call_fn_3601573?
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
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_3601612
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_3601651?
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
"__inference__wrapped_model_3596710?ECDHFG$%./899?6
/?,
*?'
gru_56_input?????????
? "7?4
2
dense_65&?#
dense_65??????????
E__inference_dense_63_layer_call_and_return_conditional_losses_3601306f$%4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
*__inference_dense_63_layer_call_fn_3601275Y$%4?1
*?'
%?"
inputs??????????
? "????????????
E__inference_dense_64_layer_call_and_return_conditional_losses_3601373f./4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
*__inference_dense_64_layer_call_fn_3601342Y./4?1
*?'
%?"
inputs??????????
? "????????????
E__inference_dense_65_layer_call_and_return_conditional_losses_3601439e894?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
*__inference_dense_65_layer_call_fn_3601409X894?1
*?'
%?"
inputs??????????
? "???????????
G__inference_dropout_91_layer_call_and_return_conditional_losses_3600571f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_91_layer_call_and_return_conditional_losses_3600583f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_91_layer_call_fn_3600561Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_91_layer_call_fn_3600566Y8?5
.?+
%?"
inputs??????????
p
? "????????????
G__inference_dropout_92_layer_call_and_return_conditional_losses_3601254f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_92_layer_call_and_return_conditional_losses_3601266f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_92_layer_call_fn_3601244Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_92_layer_call_fn_3601249Y8?5
.?+
%?"
inputs??????????
p
? "????????????
G__inference_dropout_93_layer_call_and_return_conditional_losses_3601321f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_93_layer_call_and_return_conditional_losses_3601333f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_93_layer_call_fn_3601311Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_93_layer_call_fn_3601316Y8?5
.?+
%?"
inputs??????????
p
? "????????????
G__inference_dropout_94_layer_call_and_return_conditional_losses_3601388f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_94_layer_call_and_return_conditional_losses_3601400f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_94_layer_call_fn_3601378Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_94_layer_call_fn_3601383Y8?5
.?+
%?"
inputs??????????
p
? "????????????
C__inference_gru_56_layer_call_and_return_conditional_losses_3600097?ECDO?L
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
C__inference_gru_56_layer_call_and_return_conditional_losses_3600250?ECDO?L
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
C__inference_gru_56_layer_call_and_return_conditional_losses_3600403rECD??<
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
C__inference_gru_56_layer_call_and_return_conditional_losses_3600556rECD??<
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
(__inference_gru_56_layer_call_fn_3599911~ECDO?L
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
(__inference_gru_56_layer_call_fn_3599922~ECDO?L
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
(__inference_gru_56_layer_call_fn_3599933eECD??<
5?2
$?!
inputs?????????

 
p 

 
? "????????????
(__inference_gru_56_layer_call_fn_3599944eECD??<
5?2
$?!
inputs?????????

 
p

 
? "????????????
C__inference_gru_57_layer_call_and_return_conditional_losses_3600780?HFGP?M
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
C__inference_gru_57_layer_call_and_return_conditional_losses_3600933?HFGP?M
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
C__inference_gru_57_layer_call_and_return_conditional_losses_3601086sHFG@?=
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
C__inference_gru_57_layer_call_and_return_conditional_losses_3601239sHFG@?=
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
(__inference_gru_57_layer_call_fn_3600594HFGP?M
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
(__inference_gru_57_layer_call_fn_3600605HFGP?M
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
(__inference_gru_57_layer_call_fn_3600616fHFG@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
(__inference_gru_57_layer_call_fn_3600627fHFG@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_3601506?ECD]?Z
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
H__inference_gru_cell_56_layer_call_and_return_conditional_losses_3601545?ECD]?Z
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
-__inference_gru_cell_56_layer_call_fn_3601453?ECD]?Z
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
-__inference_gru_cell_56_layer_call_fn_3601467?ECD]?Z
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
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_3601612?HFG^?[
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
H__inference_gru_cell_57_layer_call_and_return_conditional_losses_3601651?HFG^?[
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
-__inference_gru_cell_57_layer_call_fn_3601559?HFG^?[
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
-__inference_gru_cell_57_layer_call_fn_3601573?HFG^?[
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
J__inference_sequential_28_layer_call_and_return_conditional_losses_3598968|ECDHFG$%./89A?>
7?4
*?'
gru_56_input?????????
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_28_layer_call_and_return_conditional_losses_3599005|ECDHFG$%./89A?>
7?4
*?'
gru_56_input?????????
p

 
? ")?&
?
0?????????
? ?
J__inference_sequential_28_layer_call_and_return_conditional_losses_3599486vECDHFG$%./89;?8
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
J__inference_sequential_28_layer_call_and_return_conditional_losses_3599900vECDHFG$%./89;?8
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
/__inference_sequential_28_layer_call_fn_3598335oECDHFG$%./89A?>
7?4
*?'
gru_56_input?????????
p 

 
? "???????????
/__inference_sequential_28_layer_call_fn_3598931oECDHFG$%./89A?>
7?4
*?'
gru_56_input?????????
p

 
? "???????????
/__inference_sequential_28_layer_call_fn_3599071iECDHFG$%./89;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
/__inference_sequential_28_layer_call_fn_3599100iECDHFG$%./89;?8
1?.
$?!
inputs?????????
p

 
? "???????????
%__inference_signature_wrapper_3599042?ECDHFG$%./89I?F
? 
??<
:
gru_56_input*?'
gru_56_input?????????"7?4
2
dense_65&?#
dense_65?????????