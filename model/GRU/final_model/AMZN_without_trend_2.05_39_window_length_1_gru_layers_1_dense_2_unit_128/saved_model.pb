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
dense_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_42/kernel
u
#dense_42/kernel/Read/ReadVariableOpReadVariableOpdense_42/kernel* 
_output_shapes
:
??*
dtype0
s
dense_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_42/bias
l
!dense_42/bias/Read/ReadVariableOpReadVariableOpdense_42/bias*
_output_shapes	
:?*
dtype0
|
dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_43/kernel
u
#dense_43/kernel/Read/ReadVariableOpReadVariableOpdense_43/kernel* 
_output_shapes
:
??*
dtype0
s
dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_43/bias
l
!dense_43/bias/Read/ReadVariableOpReadVariableOpdense_43/bias*
_output_shapes	
:?*
dtype0
{
dense_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_44/kernel
t
#dense_44/kernel/Read/ReadVariableOpReadVariableOpdense_44/kernel*
_output_shapes
:	?*
dtype0
r
dense_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_44/bias
k
!dense_44/bias/Read/ReadVariableOpReadVariableOpdense_44/bias*
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
gru_42/gru_cell_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namegru_42/gru_cell_42/kernel
?
-gru_42/gru_cell_42/kernel/Read/ReadVariableOpReadVariableOpgru_42/gru_cell_42/kernel*
_output_shapes
:	?*
dtype0
?
#gru_42/gru_cell_42/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#gru_42/gru_cell_42/recurrent_kernel
?
7gru_42/gru_cell_42/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_42/gru_cell_42/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
gru_42/gru_cell_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_namegru_42/gru_cell_42/bias
?
+gru_42/gru_cell_42/bias/Read/ReadVariableOpReadVariableOpgru_42/gru_cell_42/bias*
_output_shapes
:	?*
dtype0
?
gru_43/gru_cell_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_namegru_43/gru_cell_43/kernel
?
-gru_43/gru_cell_43/kernel/Read/ReadVariableOpReadVariableOpgru_43/gru_cell_43/kernel* 
_output_shapes
:
??*
dtype0
?
#gru_43/gru_cell_43/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#gru_43/gru_cell_43/recurrent_kernel
?
7gru_43/gru_cell_43/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_43/gru_cell_43/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
gru_43/gru_cell_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_namegru_43/gru_cell_43/bias
?
+gru_43/gru_cell_43/bias/Read/ReadVariableOpReadVariableOpgru_43/gru_cell_43/bias*
_output_shapes
:	?*
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
Adam/dense_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_42/kernel/m
?
*Adam/dense_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_42/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_42/bias/m
z
(Adam/dense_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_42/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_43/kernel/m
?
*Adam/dense_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_43/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_43/bias/m
z
(Adam/dense_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_43/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_44/kernel/m
?
*Adam/dense_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_44/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_44/bias/m
y
(Adam/dense_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_44/bias/m*
_output_shapes
:*
dtype0
?
 Adam/gru_42/gru_cell_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/gru_42/gru_cell_42/kernel/m
?
4Adam/gru_42/gru_cell_42/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_42/gru_cell_42/kernel/m*
_output_shapes
:	?*
dtype0
?
*Adam/gru_42/gru_cell_42/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_42/gru_cell_42/recurrent_kernel/m
?
>Adam/gru_42/gru_cell_42/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_42/gru_cell_42/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/gru_42/gru_cell_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_42/gru_cell_42/bias/m
?
2Adam/gru_42/gru_cell_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_42/gru_cell_42/bias/m*
_output_shapes
:	?*
dtype0
?
 Adam/gru_43/gru_cell_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/gru_43/gru_cell_43/kernel/m
?
4Adam/gru_43/gru_cell_43/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_43/gru_cell_43/kernel/m* 
_output_shapes
:
??*
dtype0
?
*Adam/gru_43/gru_cell_43/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_43/gru_cell_43/recurrent_kernel/m
?
>Adam/gru_43/gru_cell_43/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_43/gru_cell_43/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/gru_43/gru_cell_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_43/gru_cell_43/bias/m
?
2Adam/gru_43/gru_cell_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_43/gru_cell_43/bias/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_42/kernel/v
?
*Adam/dense_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_42/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_42/bias/v
z
(Adam/dense_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_42/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_43/kernel/v
?
*Adam/dense_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_43/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_43/bias/v
z
(Adam/dense_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_43/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_44/kernel/v
?
*Adam/dense_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_44/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_44/bias/v
y
(Adam/dense_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_44/bias/v*
_output_shapes
:*
dtype0
?
 Adam/gru_42/gru_cell_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/gru_42/gru_cell_42/kernel/v
?
4Adam/gru_42/gru_cell_42/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_42/gru_cell_42/kernel/v*
_output_shapes
:	?*
dtype0
?
*Adam/gru_42/gru_cell_42/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_42/gru_cell_42/recurrent_kernel/v
?
>Adam/gru_42/gru_cell_42/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_42/gru_cell_42/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/gru_42/gru_cell_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_42/gru_cell_42/bias/v
?
2Adam/gru_42/gru_cell_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_42/gru_cell_42/bias/v*
_output_shapes
:	?*
dtype0
?
 Adam/gru_43/gru_cell_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" Adam/gru_43/gru_cell_43/kernel/v
?
4Adam/gru_43/gru_cell_43/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_43/gru_cell_43/kernel/v* 
_output_shapes
:
??*
dtype0
?
*Adam/gru_43/gru_cell_43/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/gru_43/gru_cell_43/recurrent_kernel/v
?
>Adam/gru_43/gru_cell_43/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_43/gru_cell_43/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/gru_43/gru_cell_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_43/gru_cell_43/bias/v
?
2Adam/gru_43/gru_cell_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_43/gru_cell_43/bias/v*
_output_shapes
:	?*
dtype0

NoOpNoOp
?N
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?M
value?MB?M B?M
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
trainable_variables
	variables
regularization_losses
	keras_api

signatures
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
R
 trainable_variables
!	variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
R
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
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
trainable_variables
Ilayer_regularization_losses
Jnon_trainable_variables
Klayer_metrics

Llayers
Mmetrics
	variables
regularization_losses
 
~

Ckernel
Drecurrent_kernel
Ebias
Ntrainable_variables
O	variables
Pregularization_losses
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
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
Tlayer_metrics

Ulayers
Vmetrics
	variables
regularization_losses

Wstates
 
 
 
?
trainable_variables
Xlayer_regularization_losses
Ynon_trainable_variables
Zlayer_metrics

[layers
\metrics
	variables
regularization_losses
~

Fkernel
Grecurrent_kernel
Hbias
]trainable_variables
^	variables
_regularization_losses
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
trainable_variables
alayer_regularization_losses
bnon_trainable_variables
clayer_metrics

dlayers
emetrics
	variables
regularization_losses

fstates
 
 
 
?
 trainable_variables
glayer_regularization_losses
hnon_trainable_variables
ilayer_metrics

jlayers
kmetrics
!	variables
"regularization_losses
[Y
VARIABLE_VALUEdense_42/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_42/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?
&trainable_variables
llayer_regularization_losses
mnon_trainable_variables
nlayer_metrics

olayers
pmetrics
'	variables
(regularization_losses
 
 
 
?
*trainable_variables
qlayer_regularization_losses
rnon_trainable_variables
slayer_metrics

tlayers
umetrics
+	variables
,regularization_losses
[Y
VARIABLE_VALUEdense_43/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_43/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
?
0trainable_variables
vlayer_regularization_losses
wnon_trainable_variables
xlayer_metrics

ylayers
zmetrics
1	variables
2regularization_losses
 
 
 
?
4trainable_variables
{layer_regularization_losses
|non_trainable_variables
}layer_metrics

~layers
metrics
5	variables
6regularization_losses
[Y
VARIABLE_VALUEdense_44/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_44/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
?
:trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
?metrics
;	variables
<regularization_losses
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
VARIABLE_VALUEgru_42/gru_cell_42/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#gru_42/gru_cell_42/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_42/gru_cell_42/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEgru_43/gru_cell_43/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#gru_43/gru_cell_43/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_43/gru_cell_43/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
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

?0
?1
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
Ntrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
?metrics
O	variables
Pregularization_losses
 
 
 

0
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

F0
G1
H2
 
?
]trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
?metrics
^	variables
_regularization_losses
 
 
 
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
VARIABLE_VALUEAdam/dense_42/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_42/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_43/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_43/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_44/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_44/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/gru_42/gru_cell_42/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_42/gru_cell_42/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/gru_42/gru_cell_42/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/gru_43/gru_cell_43/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_43/gru_cell_43/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/gru_43/gru_cell_43/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_42/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_42/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_43/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_43/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_44/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_44/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/gru_42/gru_cell_42/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_42/gru_cell_42/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/gru_42/gru_cell_42/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/gru_43/gru_cell_43/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_43/gru_cell_43/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/gru_43/gru_cell_43/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_gru_42_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_42_inputgru_42/gru_cell_42/biasgru_42/gru_cell_42/kernel#gru_42/gru_cell_42/recurrent_kernelgru_43/gru_cell_43/biasgru_43/gru_cell_43/kernel#gru_43/gru_cell_43/recurrent_kerneldense_42/kerneldense_42/biasdense_43/kerneldense_43/biasdense_44/kerneldense_44/bias*
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
%__inference_signature_wrapper_1574248
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_42/kernel/Read/ReadVariableOp!dense_42/bias/Read/ReadVariableOp#dense_43/kernel/Read/ReadVariableOp!dense_43/bias/Read/ReadVariableOp#dense_44/kernel/Read/ReadVariableOp!dense_44/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-gru_42/gru_cell_42/kernel/Read/ReadVariableOp7gru_42/gru_cell_42/recurrent_kernel/Read/ReadVariableOp+gru_42/gru_cell_42/bias/Read/ReadVariableOp-gru_43/gru_cell_43/kernel/Read/ReadVariableOp7gru_43/gru_cell_43/recurrent_kernel/Read/ReadVariableOp+gru_43/gru_cell_43/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_42/kernel/m/Read/ReadVariableOp(Adam/dense_42/bias/m/Read/ReadVariableOp*Adam/dense_43/kernel/m/Read/ReadVariableOp(Adam/dense_43/bias/m/Read/ReadVariableOp*Adam/dense_44/kernel/m/Read/ReadVariableOp(Adam/dense_44/bias/m/Read/ReadVariableOp4Adam/gru_42/gru_cell_42/kernel/m/Read/ReadVariableOp>Adam/gru_42/gru_cell_42/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_42/gru_cell_42/bias/m/Read/ReadVariableOp4Adam/gru_43/gru_cell_43/kernel/m/Read/ReadVariableOp>Adam/gru_43/gru_cell_43/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_43/gru_cell_43/bias/m/Read/ReadVariableOp*Adam/dense_42/kernel/v/Read/ReadVariableOp(Adam/dense_42/bias/v/Read/ReadVariableOp*Adam/dense_43/kernel/v/Read/ReadVariableOp(Adam/dense_43/bias/v/Read/ReadVariableOp*Adam/dense_44/kernel/v/Read/ReadVariableOp(Adam/dense_44/bias/v/Read/ReadVariableOp4Adam/gru_42/gru_cell_42/kernel/v/Read/ReadVariableOp>Adam/gru_42/gru_cell_42/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_42/gru_cell_42/bias/v/Read/ReadVariableOp4Adam/gru_43/gru_cell_43/kernel/v/Read/ReadVariableOp>Adam/gru_43/gru_cell_43/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_43/gru_cell_43/bias/v/Read/ReadVariableOpConst*:
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
 __inference__traced_save_1577015
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_42/kerneldense_42/biasdense_43/kerneldense_43/biasdense_44/kerneldense_44/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_42/gru_cell_42/kernel#gru_42/gru_cell_42/recurrent_kernelgru_42/gru_cell_42/biasgru_43/gru_cell_43/kernel#gru_43/gru_cell_43/recurrent_kernelgru_43/gru_cell_43/biastotalcounttotal_1count_1Adam/dense_42/kernel/mAdam/dense_42/bias/mAdam/dense_43/kernel/mAdam/dense_43/bias/mAdam/dense_44/kernel/mAdam/dense_44/bias/m Adam/gru_42/gru_cell_42/kernel/m*Adam/gru_42/gru_cell_42/recurrent_kernel/mAdam/gru_42/gru_cell_42/bias/m Adam/gru_43/gru_cell_43/kernel/m*Adam/gru_43/gru_cell_43/recurrent_kernel/mAdam/gru_43/gru_cell_43/bias/mAdam/dense_42/kernel/vAdam/dense_42/bias/vAdam/dense_43/kernel/vAdam/dense_43/bias/vAdam/dense_44/kernel/vAdam/dense_44/bias/v Adam/gru_42/gru_cell_42/kernel/v*Adam/gru_42/gru_cell_42/recurrent_kernel/vAdam/gru_42/gru_cell_42/bias/v Adam/gru_43/gru_cell_43/kernel/v*Adam/gru_43/gru_cell_43/recurrent_kernel/vAdam/gru_43/gru_cell_43/bias/v*9
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
#__inference__traced_restore_1577160??)
?
?
/__inference_sequential_21_layer_call_fn_1575106

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

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
J__inference_sequential_21_layer_call_and_return_conditional_losses_15740812
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
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?P
?	
gru_42_while_body_1574312*
&gru_42_while_gru_42_while_loop_counter0
,gru_42_while_gru_42_while_maximum_iterations
gru_42_while_placeholder
gru_42_while_placeholder_1
gru_42_while_placeholder_2)
%gru_42_while_gru_42_strided_slice_1_0e
agru_42_while_tensorarrayv2read_tensorlistgetitem_gru_42_tensorarrayunstack_tensorlistfromtensor_0E
2gru_42_while_gru_cell_42_readvariableop_resource_0:	?L
9gru_42_while_gru_cell_42_matmul_readvariableop_resource_0:	?O
;gru_42_while_gru_cell_42_matmul_1_readvariableop_resource_0:
??
gru_42_while_identity
gru_42_while_identity_1
gru_42_while_identity_2
gru_42_while_identity_3
gru_42_while_identity_4'
#gru_42_while_gru_42_strided_slice_1c
_gru_42_while_tensorarrayv2read_tensorlistgetitem_gru_42_tensorarrayunstack_tensorlistfromtensorC
0gru_42_while_gru_cell_42_readvariableop_resource:	?J
7gru_42_while_gru_cell_42_matmul_readvariableop_resource:	?M
9gru_42_while_gru_cell_42_matmul_1_readvariableop_resource:
????.gru_42/while/gru_cell_42/MatMul/ReadVariableOp?0gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp?'gru_42/while/gru_cell_42/ReadVariableOp?
>gru_42/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_42/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_42/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_42_while_tensorarrayv2read_tensorlistgetitem_gru_42_tensorarrayunstack_tensorlistfromtensor_0gru_42_while_placeholderGgru_42/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_42/while/TensorArrayV2Read/TensorListGetItem?
'gru_42/while/gru_cell_42/ReadVariableOpReadVariableOp2gru_42_while_gru_cell_42_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_42/while/gru_cell_42/ReadVariableOp?
 gru_42/while/gru_cell_42/unstackUnpack/gru_42/while/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_42/while/gru_cell_42/unstack?
.gru_42/while/gru_cell_42/MatMul/ReadVariableOpReadVariableOp9gru_42_while_gru_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_42/while/gru_cell_42/MatMul/ReadVariableOp?
gru_42/while/gru_cell_42/MatMulMatMul7gru_42/while/TensorArrayV2Read/TensorListGetItem:item:06gru_42/while/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_42/while/gru_cell_42/MatMul?
 gru_42/while/gru_cell_42/BiasAddBiasAdd)gru_42/while/gru_cell_42/MatMul:product:0)gru_42/while/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_42/while/gru_cell_42/BiasAdd?
(gru_42/while/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_42/while/gru_cell_42/split/split_dim?
gru_42/while/gru_cell_42/splitSplit1gru_42/while/gru_cell_42/split/split_dim:output:0)gru_42/while/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_42/while/gru_cell_42/split?
0gru_42/while/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp;gru_42_while_gru_cell_42_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp?
!gru_42/while/gru_cell_42/MatMul_1MatMulgru_42_while_placeholder_28gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_42/while/gru_cell_42/MatMul_1?
"gru_42/while/gru_cell_42/BiasAdd_1BiasAdd+gru_42/while/gru_cell_42/MatMul_1:product:0)gru_42/while/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_42/while/gru_cell_42/BiasAdd_1?
gru_42/while/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2 
gru_42/while/gru_cell_42/Const?
*gru_42/while/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_42/while/gru_cell_42/split_1/split_dim?
 gru_42/while/gru_cell_42/split_1SplitV+gru_42/while/gru_cell_42/BiasAdd_1:output:0'gru_42/while/gru_cell_42/Const:output:03gru_42/while/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_42/while/gru_cell_42/split_1?
gru_42/while/gru_cell_42/addAddV2'gru_42/while/gru_cell_42/split:output:0)gru_42/while/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_42/while/gru_cell_42/add?
 gru_42/while/gru_cell_42/SigmoidSigmoid gru_42/while/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_42/while/gru_cell_42/Sigmoid?
gru_42/while/gru_cell_42/add_1AddV2'gru_42/while/gru_cell_42/split:output:1)gru_42/while/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_42/while/gru_cell_42/add_1?
"gru_42/while/gru_cell_42/Sigmoid_1Sigmoid"gru_42/while/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_42/while/gru_cell_42/Sigmoid_1?
gru_42/while/gru_cell_42/mulMul&gru_42/while/gru_cell_42/Sigmoid_1:y:0)gru_42/while/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_42/while/gru_cell_42/mul?
gru_42/while/gru_cell_42/add_2AddV2'gru_42/while/gru_cell_42/split:output:2 gru_42/while/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_42/while/gru_cell_42/add_2?
gru_42/while/gru_cell_42/ReluRelu"gru_42/while/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_42/while/gru_cell_42/Relu?
gru_42/while/gru_cell_42/mul_1Mul$gru_42/while/gru_cell_42/Sigmoid:y:0gru_42_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_42/while/gru_cell_42/mul_1?
gru_42/while/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_42/while/gru_cell_42/sub/x?
gru_42/while/gru_cell_42/subSub'gru_42/while/gru_cell_42/sub/x:output:0$gru_42/while/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_42/while/gru_cell_42/sub?
gru_42/while/gru_cell_42/mul_2Mul gru_42/while/gru_cell_42/sub:z:0+gru_42/while/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_42/while/gru_cell_42/mul_2?
gru_42/while/gru_cell_42/add_3AddV2"gru_42/while/gru_cell_42/mul_1:z:0"gru_42/while/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_42/while/gru_cell_42/add_3?
1gru_42/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_42_while_placeholder_1gru_42_while_placeholder"gru_42/while/gru_cell_42/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_42/while/TensorArrayV2Write/TensorListSetItemj
gru_42/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_42/while/add/y?
gru_42/while/addAddV2gru_42_while_placeholdergru_42/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_42/while/addn
gru_42/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_42/while/add_1/y?
gru_42/while/add_1AddV2&gru_42_while_gru_42_while_loop_countergru_42/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_42/while/add_1?
gru_42/while/IdentityIdentitygru_42/while/add_1:z:0^gru_42/while/NoOp*
T0*
_output_shapes
: 2
gru_42/while/Identity?
gru_42/while/Identity_1Identity,gru_42_while_gru_42_while_maximum_iterations^gru_42/while/NoOp*
T0*
_output_shapes
: 2
gru_42/while/Identity_1?
gru_42/while/Identity_2Identitygru_42/while/add:z:0^gru_42/while/NoOp*
T0*
_output_shapes
: 2
gru_42/while/Identity_2?
gru_42/while/Identity_3IdentityAgru_42/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_42/while/NoOp*
T0*
_output_shapes
: 2
gru_42/while/Identity_3?
gru_42/while/Identity_4Identity"gru_42/while/gru_cell_42/add_3:z:0^gru_42/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_42/while/Identity_4?
gru_42/while/NoOpNoOp/^gru_42/while/gru_cell_42/MatMul/ReadVariableOp1^gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp(^gru_42/while/gru_cell_42/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_42/while/NoOp"L
#gru_42_while_gru_42_strided_slice_1%gru_42_while_gru_42_strided_slice_1_0"x
9gru_42_while_gru_cell_42_matmul_1_readvariableop_resource;gru_42_while_gru_cell_42_matmul_1_readvariableop_resource_0"t
7gru_42_while_gru_cell_42_matmul_readvariableop_resource9gru_42_while_gru_cell_42_matmul_readvariableop_resource_0"f
0gru_42_while_gru_cell_42_readvariableop_resource2gru_42_while_gru_cell_42_readvariableop_resource_0"7
gru_42_while_identitygru_42/while/Identity:output:0";
gru_42_while_identity_1 gru_42/while/Identity_1:output:0";
gru_42_while_identity_2 gru_42/while/Identity_2:output:0";
gru_42_while_identity_3 gru_42/while/Identity_3:output:0";
gru_42_while_identity_4 gru_42/while/Identity_4:output:0"?
_gru_42_while_tensorarrayv2read_tensorlistgetitem_gru_42_tensorarrayunstack_tensorlistfromtensoragru_42_while_tensorarrayv2read_tensorlistgetitem_gru_42_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_42/while/gru_cell_42/MatMul/ReadVariableOp.gru_42/while/gru_cell_42/MatMul/ReadVariableOp2d
0gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp0gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp2R
'gru_42/while/gru_cell_42/ReadVariableOp'gru_42/while/gru_cell_42/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?X
?
C__inference_gru_43_layer_call_and_return_conditional_losses_1573806

inputs6
#gru_cell_43_readvariableop_resource:	?>
*gru_cell_43_matmul_readvariableop_resource:
??@
,gru_cell_43_matmul_1_readvariableop_resource:
??
identity??!gru_cell_43/MatMul/ReadVariableOp?#gru_cell_43/MatMul_1/ReadVariableOp?gru_cell_43/ReadVariableOp?whileD
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
B :?2
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
:??????????2
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
:??????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_43/ReadVariableOpReadVariableOp#gru_cell_43_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_43/ReadVariableOp?
gru_cell_43/unstackUnpack"gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_43/unstack?
!gru_cell_43/MatMul/ReadVariableOpReadVariableOp*gru_cell_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_43/MatMul/ReadVariableOp?
gru_cell_43/MatMulMatMulstrided_slice_2:output:0)gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul?
gru_cell_43/BiasAddBiasAddgru_cell_43/MatMul:product:0gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd?
gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split/split_dim?
gru_cell_43/splitSplit$gru_cell_43/split/split_dim:output:0gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split?
#gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_43_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_43/MatMul_1/ReadVariableOp?
gru_cell_43/MatMul_1MatMulzeros:output:0+gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul_1?
gru_cell_43/BiasAdd_1BiasAddgru_cell_43/MatMul_1:product:0gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd_1{
gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_43/Const?
gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split_1/split_dim?
gru_cell_43/split_1SplitVgru_cell_43/BiasAdd_1:output:0gru_cell_43/Const:output:0&gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split_1?
gru_cell_43/addAddV2gru_cell_43/split:output:0gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add}
gru_cell_43/SigmoidSigmoidgru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid?
gru_cell_43/add_1AddV2gru_cell_43/split:output:1gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_1?
gru_cell_43/Sigmoid_1Sigmoidgru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid_1?
gru_cell_43/mulMulgru_cell_43/Sigmoid_1:y:0gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul?
gru_cell_43/add_2AddV2gru_cell_43/split:output:2gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_2v
gru_cell_43/ReluRelugru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Relu?
gru_cell_43/mul_1Mulgru_cell_43/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_1k
gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_43/sub/x?
gru_cell_43/subSubgru_cell_43/sub/x:output:0gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/sub?
gru_cell_43/mul_2Mulgru_cell_43/sub:z:0gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_2?
gru_cell_43/add_3AddV2gru_cell_43/mul_1:z:0gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_43_readvariableop_resource*gru_cell_43_matmul_readvariableop_resource,gru_cell_43_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1573717*
condR
while_cond_1573716*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
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
:??????????2
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
:??????????2

Identity?
NoOpNoOp"^gru_cell_43/MatMul/ReadVariableOp$^gru_cell_43/MatMul_1/ReadVariableOp^gru_cell_43/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_43/MatMul/ReadVariableOp!gru_cell_43/MatMul/ReadVariableOp2J
#gru_cell_43/MatMul_1/ReadVariableOp#gru_cell_43/MatMul_1/ReadVariableOp28
gru_cell_43/ReadVariableOpgru_cell_43/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
gru_42_while_cond_1574697*
&gru_42_while_gru_42_while_loop_counter0
,gru_42_while_gru_42_while_maximum_iterations
gru_42_while_placeholder
gru_42_while_placeholder_1
gru_42_while_placeholder_2,
(gru_42_while_less_gru_42_strided_slice_1C
?gru_42_while_gru_42_while_cond_1574697___redundant_placeholder0C
?gru_42_while_gru_42_while_cond_1574697___redundant_placeholder1C
?gru_42_while_gru_42_while_cond_1574697___redundant_placeholder2C
?gru_42_while_gru_42_while_cond_1574697___redundant_placeholder3
gru_42_while_identity
?
gru_42/while/LessLessgru_42_while_placeholder(gru_42_while_less_gru_42_strided_slice_1*
T0*
_output_shapes
: 2
gru_42/while/Lessr
gru_42/while/IdentityIdentitygru_42/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_42/while/Identity"7
gru_42_while_identitygru_42/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?;
?
C__inference_gru_43_layer_call_and_return_conditional_losses_1572629

inputs&
gru_cell_43_1572553:	?'
gru_cell_43_1572555:
??'
gru_cell_43_1572557:
??
identity??#gru_cell_43/StatefulPartitionedCall?whileD
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
B :?2
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
:??????????2
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
!:???????????????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_43_1572553gru_cell_43_1572555gru_cell_43_1572557*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_15725522%
#gru_cell_43/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_43_1572553gru_cell_43_1572555gru_cell_43_1572557*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1572565*
condR
while_cond_1572564*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identity|
NoOpNoOp$^gru_cell_43/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#gru_cell_43/StatefulPartitionedCall#gru_cell_43/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?1
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1574211
gru_42_input!
gru_42_1574177:	?!
gru_42_1574179:	?"
gru_42_1574181:
??!
gru_43_1574185:	?"
gru_43_1574187:
??"
gru_43_1574189:
??$
dense_42_1574193:
??
dense_42_1574195:	?$
dense_43_1574199:
??
dense_43_1574201:	?#
dense_44_1574205:	?
dense_44_1574207:
identity?? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall?"dropout_63/StatefulPartitionedCall?"dropout_64/StatefulPartitionedCall?"dropout_65/StatefulPartitionedCall?"dropout_66/StatefulPartitionedCall?gru_42/StatefulPartitionedCall?gru_43/StatefulPartitionedCall?
gru_42/StatefulPartitionedCallStatefulPartitionedCallgru_42_inputgru_42_1574177gru_42_1574179gru_42_1574181*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_42_layer_call_and_return_conditional_losses_15740042 
gru_42/StatefulPartitionedCall?
"dropout_63/StatefulPartitionedCallStatefulPartitionedCall'gru_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_63_layer_call_and_return_conditional_losses_15738352$
"dropout_63/StatefulPartitionedCall?
gru_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_63/StatefulPartitionedCall:output:0gru_43_1574185gru_43_1574187gru_43_1574189*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_43_layer_call_and_return_conditional_losses_15738062 
gru_43/StatefulPartitionedCall?
"dropout_64/StatefulPartitionedCallStatefulPartitionedCall'gru_43/StatefulPartitionedCall:output:0#^dropout_63/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_64_layer_call_and_return_conditional_losses_15736372$
"dropout_64/StatefulPartitionedCall?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall+dropout_64/StatefulPartitionedCall:output:0dense_42_1574193dense_42_1574195*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_15734202"
 dense_42/StatefulPartitionedCall?
"dropout_65/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0#^dropout_64/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_65_layer_call_and_return_conditional_losses_15736042$
"dropout_65/StatefulPartitionedCall?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_65/StatefulPartitionedCall:output:0dense_43_1574199dense_43_1574201*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_15734642"
 dense_43/StatefulPartitionedCall?
"dropout_66/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:0#^dropout_65/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_66_layer_call_and_return_conditional_losses_15735712$
"dropout_66/StatefulPartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall+dropout_66/StatefulPartitionedCall:output:0dense_44_1574205dense_44_1574207*
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
E__inference_dense_44_layer_call_and_return_conditional_losses_15735072"
 dense_44/StatefulPartitionedCall?
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall#^dropout_63/StatefulPartitionedCall#^dropout_64/StatefulPartitionedCall#^dropout_65/StatefulPartitionedCall#^dropout_66/StatefulPartitionedCall^gru_42/StatefulPartitionedCall^gru_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2H
"dropout_63/StatefulPartitionedCall"dropout_63/StatefulPartitionedCall2H
"dropout_64/StatefulPartitionedCall"dropout_64/StatefulPartitionedCall2H
"dropout_65/StatefulPartitionedCall"dropout_65/StatefulPartitionedCall2H
"dropout_66/StatefulPartitionedCall"dropout_66/StatefulPartitionedCall2@
gru_42/StatefulPartitionedCallgru_42/StatefulPartitionedCall2@
gru_43/StatefulPartitionedCallgru_43/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_42_input
?Y
?
C__inference_gru_43_layer_call_and_return_conditional_losses_1575942
inputs_06
#gru_cell_43_readvariableop_resource:	?>
*gru_cell_43_matmul_readvariableop_resource:
??@
,gru_cell_43_matmul_1_readvariableop_resource:
??
identity??!gru_cell_43/MatMul/ReadVariableOp?#gru_cell_43/MatMul_1/ReadVariableOp?gru_cell_43/ReadVariableOp?whileF
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
B :?2
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
:??????????2
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
!:???????????????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_43/ReadVariableOpReadVariableOp#gru_cell_43_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_43/ReadVariableOp?
gru_cell_43/unstackUnpack"gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_43/unstack?
!gru_cell_43/MatMul/ReadVariableOpReadVariableOp*gru_cell_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_43/MatMul/ReadVariableOp?
gru_cell_43/MatMulMatMulstrided_slice_2:output:0)gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul?
gru_cell_43/BiasAddBiasAddgru_cell_43/MatMul:product:0gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd?
gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split/split_dim?
gru_cell_43/splitSplit$gru_cell_43/split/split_dim:output:0gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split?
#gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_43_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_43/MatMul_1/ReadVariableOp?
gru_cell_43/MatMul_1MatMulzeros:output:0+gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul_1?
gru_cell_43/BiasAdd_1BiasAddgru_cell_43/MatMul_1:product:0gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd_1{
gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_43/Const?
gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split_1/split_dim?
gru_cell_43/split_1SplitVgru_cell_43/BiasAdd_1:output:0gru_cell_43/Const:output:0&gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split_1?
gru_cell_43/addAddV2gru_cell_43/split:output:0gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add}
gru_cell_43/SigmoidSigmoidgru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid?
gru_cell_43/add_1AddV2gru_cell_43/split:output:1gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_1?
gru_cell_43/Sigmoid_1Sigmoidgru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid_1?
gru_cell_43/mulMulgru_cell_43/Sigmoid_1:y:0gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul?
gru_cell_43/add_2AddV2gru_cell_43/split:output:2gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_2v
gru_cell_43/ReluRelugru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Relu?
gru_cell_43/mul_1Mulgru_cell_43/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_1k
gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_43/sub/x?
gru_cell_43/subSubgru_cell_43/sub/x:output:0gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/sub?
gru_cell_43/mul_2Mulgru_cell_43/sub:z:0gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_2?
gru_cell_43/add_3AddV2gru_cell_43/mul_1:z:0gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_43_readvariableop_resource*gru_cell_43_matmul_readvariableop_resource,gru_cell_43_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1575853*
condR
while_cond_1575852*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identity?
NoOpNoOp"^gru_cell_43/MatMul/ReadVariableOp$^gru_cell_43/MatMul_1/ReadVariableOp^gru_cell_43/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2F
!gru_cell_43/MatMul/ReadVariableOp!gru_cell_43/MatMul/ReadVariableOp2J
#gru_cell_43/MatMul_1/ReadVariableOp#gru_cell_43/MatMul_1/ReadVariableOp28
gru_cell_43/ReadVariableOpgru_cell_43/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?E
?
while_body_1575170
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_42_readvariableop_resource_0:	?E
2while_gru_cell_42_matmul_readvariableop_resource_0:	?H
4while_gru_cell_42_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_42_readvariableop_resource:	?C
0while_gru_cell_42_matmul_readvariableop_resource:	?F
2while_gru_cell_42_matmul_1_readvariableop_resource:
????'while/gru_cell_42/MatMul/ReadVariableOp?)while/gru_cell_42/MatMul_1/ReadVariableOp? while/gru_cell_42/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_42/ReadVariableOpReadVariableOp+while_gru_cell_42_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_42/ReadVariableOp?
while/gru_cell_42/unstackUnpack(while/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_42/unstack?
'while/gru_cell_42/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_42/MatMul/ReadVariableOp?
while/gru_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul?
while/gru_cell_42/BiasAddBiasAdd"while/gru_cell_42/MatMul:product:0"while/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd?
!while/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_42/split/split_dim?
while/gru_cell_42/splitSplit*while/gru_cell_42/split/split_dim:output:0"while/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split?
)while/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_42_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_42/MatMul_1/ReadVariableOp?
while/gru_cell_42/MatMul_1MatMulwhile_placeholder_21while/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul_1?
while/gru_cell_42/BiasAdd_1BiasAdd$while/gru_cell_42/MatMul_1:product:0"while/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd_1?
while/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_42/Const?
#while/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_42/split_1/split_dim?
while/gru_cell_42/split_1SplitV$while/gru_cell_42/BiasAdd_1:output:0 while/gru_cell_42/Const:output:0,while/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split_1?
while/gru_cell_42/addAddV2 while/gru_cell_42/split:output:0"while/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add?
while/gru_cell_42/SigmoidSigmoidwhile/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid?
while/gru_cell_42/add_1AddV2 while/gru_cell_42/split:output:1"while/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_1?
while/gru_cell_42/Sigmoid_1Sigmoidwhile/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid_1?
while/gru_cell_42/mulMulwhile/gru_cell_42/Sigmoid_1:y:0"while/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul?
while/gru_cell_42/add_2AddV2 while/gru_cell_42/split:output:2while/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_2?
while/gru_cell_42/ReluReluwhile/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Relu?
while/gru_cell_42/mul_1Mulwhile/gru_cell_42/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_1w
while/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_42/sub/x?
while/gru_cell_42/subSub while/gru_cell_42/sub/x:output:0while/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/sub?
while/gru_cell_42/mul_2Mulwhile/gru_cell_42/sub:z:0$while/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_2?
while/gru_cell_42/add_3AddV2while/gru_cell_42/mul_1:z:0while/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_42/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_42/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_42/MatMul/ReadVariableOp*^while/gru_cell_42/MatMul_1/ReadVariableOp!^while/gru_cell_42/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_42_matmul_1_readvariableop_resource4while_gru_cell_42_matmul_1_readvariableop_resource_0"f
0while_gru_cell_42_matmul_readvariableop_resource2while_gru_cell_42_matmul_readvariableop_resource_0"X
)while_gru_cell_42_readvariableop_resource+while_gru_cell_42_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_42/MatMul/ReadVariableOp'while/gru_cell_42/MatMul/ReadVariableOp2V
)while/gru_cell_42/MatMul_1/ReadVariableOp)while/gru_cell_42/MatMul_1/ReadVariableOp2D
 while/gru_cell_42/ReadVariableOp while/gru_cell_42/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
-__inference_gru_cell_43_layer_call_fn_1576857

inputs
states_0
unknown:	?
	unknown_0:
??
	unknown_1:
??
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
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_15726952
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

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
.:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
'sequential_21_gru_43_while_cond_1571743F
Bsequential_21_gru_43_while_sequential_21_gru_43_while_loop_counterL
Hsequential_21_gru_43_while_sequential_21_gru_43_while_maximum_iterations*
&sequential_21_gru_43_while_placeholder,
(sequential_21_gru_43_while_placeholder_1,
(sequential_21_gru_43_while_placeholder_2H
Dsequential_21_gru_43_while_less_sequential_21_gru_43_strided_slice_1_
[sequential_21_gru_43_while_sequential_21_gru_43_while_cond_1571743___redundant_placeholder0_
[sequential_21_gru_43_while_sequential_21_gru_43_while_cond_1571743___redundant_placeholder1_
[sequential_21_gru_43_while_sequential_21_gru_43_while_cond_1571743___redundant_placeholder2_
[sequential_21_gru_43_while_sequential_21_gru_43_while_cond_1571743___redundant_placeholder3'
#sequential_21_gru_43_while_identity
?
sequential_21/gru_43/while/LessLess&sequential_21_gru_43_while_placeholderDsequential_21_gru_43_while_less_sequential_21_gru_43_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_21/gru_43/while/Less?
#sequential_21/gru_43/while/IdentityIdentity#sequential_21/gru_43/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_21/gru_43/while/Identity"S
#sequential_21_gru_43_while_identity,sequential_21/gru_43/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
(__inference_gru_42_layer_call_fn_1575729
inputs_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_42_layer_call_and_return_conditional_losses_15720632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
e
,__inference_dropout_64_layer_call_fn_1576472

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_64_layer_call_and_return_conditional_losses_15736372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?;
?
C__inference_gru_43_layer_call_and_return_conditional_losses_1572822

inputs&
gru_cell_43_1572746:	?'
gru_cell_43_1572748:
??'
gru_cell_43_1572750:
??
identity??#gru_cell_43/StatefulPartitionedCall?whileD
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
B :?2
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
:??????????2
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
!:???????????????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_43_1572746gru_cell_43_1572748gru_cell_43_1572750*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_15726952%
#gru_cell_43/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_43_1572746gru_cell_43_1572748gru_cell_43_1572750*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1572758*
condR
while_cond_1572757*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identity|
NoOpNoOp$^gru_cell_43/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#gru_cell_43/StatefulPartitionedCall#gru_cell_43/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?;
?
C__inference_gru_42_layer_call_and_return_conditional_losses_1572063

inputs&
gru_cell_42_1571987:	?&
gru_cell_42_1571989:	?'
gru_cell_42_1571991:
??
identity??#gru_cell_42/StatefulPartitionedCall?whileD
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
B :?2
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
:??????????2
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
 :??????????????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_42_1571987gru_cell_42_1571989gru_cell_42_1571991*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_15719862%
#gru_cell_42/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_42_1571987gru_cell_42_1571989gru_cell_42_1571991*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1571999*
condR
while_cond_1571998*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identity|
NoOpNoOp$^gru_cell_42/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#gru_cell_42/StatefulPartitionedCall#gru_cell_42/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?!
?
E__inference_dense_43_layer_call_and_return_conditional_losses_1576570

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_21_layer_call_fn_1575077

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

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
J__inference_sequential_21_layer_call_and_return_conditional_losses_15735142
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
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_64_layer_call_and_return_conditional_losses_1576450

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_gru_43_layer_call_fn_1576423
inputs_0
unknown:	?
	unknown_0:
??
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_43_layer_call_and_return_conditional_losses_15728222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_1572757
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1572757___redundant_placeholder05
1while_while_cond_1572757___redundant_placeholder15
1while_while_cond_1572757___redundant_placeholder25
1while_while_cond_1572757___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?P
?	
gru_43_while_body_1574462*
&gru_43_while_gru_43_while_loop_counter0
,gru_43_while_gru_43_while_maximum_iterations
gru_43_while_placeholder
gru_43_while_placeholder_1
gru_43_while_placeholder_2)
%gru_43_while_gru_43_strided_slice_1_0e
agru_43_while_tensorarrayv2read_tensorlistgetitem_gru_43_tensorarrayunstack_tensorlistfromtensor_0E
2gru_43_while_gru_cell_43_readvariableop_resource_0:	?M
9gru_43_while_gru_cell_43_matmul_readvariableop_resource_0:
??O
;gru_43_while_gru_cell_43_matmul_1_readvariableop_resource_0:
??
gru_43_while_identity
gru_43_while_identity_1
gru_43_while_identity_2
gru_43_while_identity_3
gru_43_while_identity_4'
#gru_43_while_gru_43_strided_slice_1c
_gru_43_while_tensorarrayv2read_tensorlistgetitem_gru_43_tensorarrayunstack_tensorlistfromtensorC
0gru_43_while_gru_cell_43_readvariableop_resource:	?K
7gru_43_while_gru_cell_43_matmul_readvariableop_resource:
??M
9gru_43_while_gru_cell_43_matmul_1_readvariableop_resource:
????.gru_43/while/gru_cell_43/MatMul/ReadVariableOp?0gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp?'gru_43/while/gru_cell_43/ReadVariableOp?
>gru_43/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>gru_43/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_43/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_43_while_tensorarrayv2read_tensorlistgetitem_gru_43_tensorarrayunstack_tensorlistfromtensor_0gru_43_while_placeholderGgru_43/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0gru_43/while/TensorArrayV2Read/TensorListGetItem?
'gru_43/while/gru_cell_43/ReadVariableOpReadVariableOp2gru_43_while_gru_cell_43_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_43/while/gru_cell_43/ReadVariableOp?
 gru_43/while/gru_cell_43/unstackUnpack/gru_43/while/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_43/while/gru_cell_43/unstack?
.gru_43/while/gru_cell_43/MatMul/ReadVariableOpReadVariableOp9gru_43_while_gru_cell_43_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype020
.gru_43/while/gru_cell_43/MatMul/ReadVariableOp?
gru_43/while/gru_cell_43/MatMulMatMul7gru_43/while/TensorArrayV2Read/TensorListGetItem:item:06gru_43/while/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_43/while/gru_cell_43/MatMul?
 gru_43/while/gru_cell_43/BiasAddBiasAdd)gru_43/while/gru_cell_43/MatMul:product:0)gru_43/while/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_43/while/gru_cell_43/BiasAdd?
(gru_43/while/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_43/while/gru_cell_43/split/split_dim?
gru_43/while/gru_cell_43/splitSplit1gru_43/while/gru_cell_43/split/split_dim:output:0)gru_43/while/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_43/while/gru_cell_43/split?
0gru_43/while/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp;gru_43_while_gru_cell_43_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp?
!gru_43/while/gru_cell_43/MatMul_1MatMulgru_43_while_placeholder_28gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_43/while/gru_cell_43/MatMul_1?
"gru_43/while/gru_cell_43/BiasAdd_1BiasAdd+gru_43/while/gru_cell_43/MatMul_1:product:0)gru_43/while/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_43/while/gru_cell_43/BiasAdd_1?
gru_43/while/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2 
gru_43/while/gru_cell_43/Const?
*gru_43/while/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_43/while/gru_cell_43/split_1/split_dim?
 gru_43/while/gru_cell_43/split_1SplitV+gru_43/while/gru_cell_43/BiasAdd_1:output:0'gru_43/while/gru_cell_43/Const:output:03gru_43/while/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_43/while/gru_cell_43/split_1?
gru_43/while/gru_cell_43/addAddV2'gru_43/while/gru_cell_43/split:output:0)gru_43/while/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_43/while/gru_cell_43/add?
 gru_43/while/gru_cell_43/SigmoidSigmoid gru_43/while/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_43/while/gru_cell_43/Sigmoid?
gru_43/while/gru_cell_43/add_1AddV2'gru_43/while/gru_cell_43/split:output:1)gru_43/while/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_43/while/gru_cell_43/add_1?
"gru_43/while/gru_cell_43/Sigmoid_1Sigmoid"gru_43/while/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_43/while/gru_cell_43/Sigmoid_1?
gru_43/while/gru_cell_43/mulMul&gru_43/while/gru_cell_43/Sigmoid_1:y:0)gru_43/while/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_43/while/gru_cell_43/mul?
gru_43/while/gru_cell_43/add_2AddV2'gru_43/while/gru_cell_43/split:output:2 gru_43/while/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_43/while/gru_cell_43/add_2?
gru_43/while/gru_cell_43/ReluRelu"gru_43/while/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_43/while/gru_cell_43/Relu?
gru_43/while/gru_cell_43/mul_1Mul$gru_43/while/gru_cell_43/Sigmoid:y:0gru_43_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_43/while/gru_cell_43/mul_1?
gru_43/while/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_43/while/gru_cell_43/sub/x?
gru_43/while/gru_cell_43/subSub'gru_43/while/gru_cell_43/sub/x:output:0$gru_43/while/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_43/while/gru_cell_43/sub?
gru_43/while/gru_cell_43/mul_2Mul gru_43/while/gru_cell_43/sub:z:0+gru_43/while/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_43/while/gru_cell_43/mul_2?
gru_43/while/gru_cell_43/add_3AddV2"gru_43/while/gru_cell_43/mul_1:z:0"gru_43/while/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_43/while/gru_cell_43/add_3?
1gru_43/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_43_while_placeholder_1gru_43_while_placeholder"gru_43/while/gru_cell_43/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_43/while/TensorArrayV2Write/TensorListSetItemj
gru_43/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_43/while/add/y?
gru_43/while/addAddV2gru_43_while_placeholdergru_43/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_43/while/addn
gru_43/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_43/while/add_1/y?
gru_43/while/add_1AddV2&gru_43_while_gru_43_while_loop_countergru_43/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_43/while/add_1?
gru_43/while/IdentityIdentitygru_43/while/add_1:z:0^gru_43/while/NoOp*
T0*
_output_shapes
: 2
gru_43/while/Identity?
gru_43/while/Identity_1Identity,gru_43_while_gru_43_while_maximum_iterations^gru_43/while/NoOp*
T0*
_output_shapes
: 2
gru_43/while/Identity_1?
gru_43/while/Identity_2Identitygru_43/while/add:z:0^gru_43/while/NoOp*
T0*
_output_shapes
: 2
gru_43/while/Identity_2?
gru_43/while/Identity_3IdentityAgru_43/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_43/while/NoOp*
T0*
_output_shapes
: 2
gru_43/while/Identity_3?
gru_43/while/Identity_4Identity"gru_43/while/gru_cell_43/add_3:z:0^gru_43/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_43/while/Identity_4?
gru_43/while/NoOpNoOp/^gru_43/while/gru_cell_43/MatMul/ReadVariableOp1^gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp(^gru_43/while/gru_cell_43/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_43/while/NoOp"L
#gru_43_while_gru_43_strided_slice_1%gru_43_while_gru_43_strided_slice_1_0"x
9gru_43_while_gru_cell_43_matmul_1_readvariableop_resource;gru_43_while_gru_cell_43_matmul_1_readvariableop_resource_0"t
7gru_43_while_gru_cell_43_matmul_readvariableop_resource9gru_43_while_gru_cell_43_matmul_readvariableop_resource_0"f
0gru_43_while_gru_cell_43_readvariableop_resource2gru_43_while_gru_cell_43_readvariableop_resource_0"7
gru_43_while_identitygru_43/while/Identity:output:0";
gru_43_while_identity_1 gru_43/while/Identity_1:output:0";
gru_43_while_identity_2 gru_43/while/Identity_2:output:0";
gru_43_while_identity_3 gru_43/while/Identity_3:output:0";
gru_43_while_identity_4 gru_43/while/Identity_4:output:0"?
_gru_43_while_tensorarrayv2read_tensorlistgetitem_gru_43_tensorarrayunstack_tensorlistfromtensoragru_43_while_tensorarrayv2read_tensorlistgetitem_gru_43_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_43/while/gru_cell_43/MatMul/ReadVariableOp.gru_43/while/gru_cell_43/MatMul/ReadVariableOp2d
0gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp0gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp2R
'gru_43/while/gru_cell_43/ReadVariableOp'gru_43/while/gru_cell_43/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1573914
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1573914___redundant_placeholder05
1while_while_cond_1573914___redundant_placeholder15
1while_while_cond_1573914___redundant_placeholder25
1while_while_cond_1573914___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?*
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1573514

inputs!
gru_42_1573208:	?!
gru_42_1573210:	?"
gru_42_1573212:
??!
gru_43_1573375:	?"
gru_43_1573377:
??"
gru_43_1573379:
??$
dense_42_1573421:
??
dense_42_1573423:	?$
dense_43_1573465:
??
dense_43_1573467:	?#
dense_44_1573508:	?
dense_44_1573510:
identity?? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall?gru_42/StatefulPartitionedCall?gru_43/StatefulPartitionedCall?
gru_42/StatefulPartitionedCallStatefulPartitionedCallinputsgru_42_1573208gru_42_1573210gru_42_1573212*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_42_layer_call_and_return_conditional_losses_15732072 
gru_42/StatefulPartitionedCall?
dropout_63/PartitionedCallPartitionedCall'gru_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_63_layer_call_and_return_conditional_losses_15732202
dropout_63/PartitionedCall?
gru_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_63/PartitionedCall:output:0gru_43_1573375gru_43_1573377gru_43_1573379*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_43_layer_call_and_return_conditional_losses_15733742 
gru_43/StatefulPartitionedCall?
dropout_64/PartitionedCallPartitionedCall'gru_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_64_layer_call_and_return_conditional_losses_15733872
dropout_64/PartitionedCall?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall#dropout_64/PartitionedCall:output:0dense_42_1573421dense_42_1573423*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_15734202"
 dense_42/StatefulPartitionedCall?
dropout_65/PartitionedCallPartitionedCall)dense_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_65_layer_call_and_return_conditional_losses_15734312
dropout_65/PartitionedCall?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_65/PartitionedCall:output:0dense_43_1573465dense_43_1573467*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_15734642"
 dense_43/StatefulPartitionedCall?
dropout_66/PartitionedCallPartitionedCall)dense_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_66_layer_call_and_return_conditional_losses_15734752
dropout_66/PartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall#dropout_66/PartitionedCall:output:0dense_44_1573508dense_44_1573510*
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
E__inference_dense_44_layer_call_and_return_conditional_losses_15735072"
 dense_44/StatefulPartitionedCall?
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall^gru_42/StatefulPartitionedCall^gru_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2@
gru_42/StatefulPartitionedCallgru_42/StatefulPartitionedCall2@
gru_43/StatefulPartitionedCallgru_43/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_gru_42_layer_call_fn_1575740
inputs_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_42_layer_call_and_return_conditional_losses_15722562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
? 
?
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_1576790

inputs
states_0*
readvariableop_resource:	?2
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
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
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
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
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
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
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

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
.:??????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
while_cond_1573117
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1573117___redundant_placeholder05
1while_while_cond_1573117___redundant_placeholder15
1while_while_cond_1573117___redundant_placeholder25
1while_while_cond_1573117___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?	
?
gru_43_while_cond_1574854*
&gru_43_while_gru_43_while_loop_counter0
,gru_43_while_gru_43_while_maximum_iterations
gru_43_while_placeholder
gru_43_while_placeholder_1
gru_43_while_placeholder_2,
(gru_43_while_less_gru_43_strided_slice_1C
?gru_43_while_gru_43_while_cond_1574854___redundant_placeholder0C
?gru_43_while_gru_43_while_cond_1574854___redundant_placeholder1C
?gru_43_while_gru_43_while_cond_1574854___redundant_placeholder2C
?gru_43_while_gru_43_while_cond_1574854___redundant_placeholder3
gru_43_while_identity
?
gru_43/while/LessLessgru_43_while_placeholder(gru_43_while_less_gru_43_strided_slice_1*
T0*
_output_shapes
: 2
gru_43/while/Lessr
gru_43/while/IdentityIdentitygru_43/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_43/while/Identity"7
gru_43_while_identitygru_43/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1576005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1576005___redundant_placeholder05
1while_while_cond_1576005___redundant_placeholder15
1while_while_cond_1576005___redundant_placeholder25
1while_while_cond_1576005___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?
e
G__inference_dropout_63_layer_call_and_return_conditional_losses_1573220

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
gru_42_while_cond_1574311*
&gru_42_while_gru_42_while_loop_counter0
,gru_42_while_gru_42_while_maximum_iterations
gru_42_while_placeholder
gru_42_while_placeholder_1
gru_42_while_placeholder_2,
(gru_42_while_less_gru_42_strided_slice_1C
?gru_42_while_gru_42_while_cond_1574311___redundant_placeholder0C
?gru_42_while_gru_42_while_cond_1574311___redundant_placeholder1C
?gru_42_while_gru_42_while_cond_1574311___redundant_placeholder2C
?gru_42_while_gru_42_while_cond_1574311___redundant_placeholder3
gru_42_while_identity
?
gru_42/while/LessLessgru_42_while_placeholder(gru_42_while_less_gru_42_strided_slice_1*
T0*
_output_shapes
: 2
gru_42/while/Lessr
gru_42/while/IdentityIdentitygru_42/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_42/while/Identity"7
gru_42_while_identitygru_42/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?X
?
C__inference_gru_42_layer_call_and_return_conditional_losses_1574004

inputs6
#gru_cell_42_readvariableop_resource:	?=
*gru_cell_42_matmul_readvariableop_resource:	?@
,gru_cell_42_matmul_1_readvariableop_resource:
??
identity??!gru_cell_42/MatMul/ReadVariableOp?#gru_cell_42/MatMul_1/ReadVariableOp?gru_cell_42/ReadVariableOp?whileD
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
B :?2
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
:??????????2
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
:?????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_42/ReadVariableOpReadVariableOp#gru_cell_42_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_42/ReadVariableOp?
gru_cell_42/unstackUnpack"gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_42/unstack?
!gru_cell_42/MatMul/ReadVariableOpReadVariableOp*gru_cell_42_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_42/MatMul/ReadVariableOp?
gru_cell_42/MatMulMatMulstrided_slice_2:output:0)gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul?
gru_cell_42/BiasAddBiasAddgru_cell_42/MatMul:product:0gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd?
gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split/split_dim?
gru_cell_42/splitSplit$gru_cell_42/split/split_dim:output:0gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split?
#gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_42_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_42/MatMul_1/ReadVariableOp?
gru_cell_42/MatMul_1MatMulzeros:output:0+gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul_1?
gru_cell_42/BiasAdd_1BiasAddgru_cell_42/MatMul_1:product:0gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd_1{
gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_42/Const?
gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split_1/split_dim?
gru_cell_42/split_1SplitVgru_cell_42/BiasAdd_1:output:0gru_cell_42/Const:output:0&gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split_1?
gru_cell_42/addAddV2gru_cell_42/split:output:0gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add}
gru_cell_42/SigmoidSigmoidgru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid?
gru_cell_42/add_1AddV2gru_cell_42/split:output:1gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_1?
gru_cell_42/Sigmoid_1Sigmoidgru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid_1?
gru_cell_42/mulMulgru_cell_42/Sigmoid_1:y:0gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul?
gru_cell_42/add_2AddV2gru_cell_42/split:output:2gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_2v
gru_cell_42/ReluRelugru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Relu?
gru_cell_42/mul_1Mulgru_cell_42/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_1k
gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_42/sub/x?
gru_cell_42/subSubgru_cell_42/sub/x:output:0gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/sub?
gru_cell_42/mul_2Mulgru_cell_42/sub:z:0gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_2?
gru_cell_42/add_3AddV2gru_cell_42/mul_1:z:0gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_42_readvariableop_resource*gru_cell_42_matmul_readvariableop_resource,gru_cell_42_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1573915*
condR
while_cond_1573914*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
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
:??????????2
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
:??????????2

Identity?
NoOpNoOp"^gru_cell_42/MatMul/ReadVariableOp$^gru_cell_42/MatMul_1/ReadVariableOp^gru_cell_42/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_42/MatMul/ReadVariableOp!gru_cell_42/MatMul/ReadVariableOp2J
#gru_cell_42/MatMul_1/ReadVariableOp#gru_cell_42/MatMul_1/ReadVariableOp28
gru_cell_42/ReadVariableOpgru_cell_42/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?

J__inference_sequential_21_layer_call_and_return_conditional_losses_1575048

inputs=
*gru_42_gru_cell_42_readvariableop_resource:	?D
1gru_42_gru_cell_42_matmul_readvariableop_resource:	?G
3gru_42_gru_cell_42_matmul_1_readvariableop_resource:
??=
*gru_43_gru_cell_43_readvariableop_resource:	?E
1gru_43_gru_cell_43_matmul_readvariableop_resource:
??G
3gru_43_gru_cell_43_matmul_1_readvariableop_resource:
??>
*dense_42_tensordot_readvariableop_resource:
??7
(dense_42_biasadd_readvariableop_resource:	?>
*dense_43_tensordot_readvariableop_resource:
??7
(dense_43_biasadd_readvariableop_resource:	?=
*dense_44_tensordot_readvariableop_resource:	?6
(dense_44_biasadd_readvariableop_resource:
identity??dense_42/BiasAdd/ReadVariableOp?!dense_42/Tensordot/ReadVariableOp?dense_43/BiasAdd/ReadVariableOp?!dense_43/Tensordot/ReadVariableOp?dense_44/BiasAdd/ReadVariableOp?!dense_44/Tensordot/ReadVariableOp?(gru_42/gru_cell_42/MatMul/ReadVariableOp?*gru_42/gru_cell_42/MatMul_1/ReadVariableOp?!gru_42/gru_cell_42/ReadVariableOp?gru_42/while?(gru_43/gru_cell_43/MatMul/ReadVariableOp?*gru_43/gru_cell_43/MatMul_1/ReadVariableOp?!gru_43/gru_cell_43/ReadVariableOp?gru_43/whileR
gru_42/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_42/Shape?
gru_42/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_42/strided_slice/stack?
gru_42/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_42/strided_slice/stack_1?
gru_42/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_42/strided_slice/stack_2?
gru_42/strided_sliceStridedSlicegru_42/Shape:output:0#gru_42/strided_slice/stack:output:0%gru_42/strided_slice/stack_1:output:0%gru_42/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_42/strided_sliceq
gru_42/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_42/zeros/packed/1?
gru_42/zeros/packedPackgru_42/strided_slice:output:0gru_42/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_42/zeros/packedm
gru_42/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_42/zeros/Const?
gru_42/zerosFillgru_42/zeros/packed:output:0gru_42/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_42/zeros?
gru_42/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_42/transpose/perm?
gru_42/transpose	Transposeinputsgru_42/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
gru_42/transposed
gru_42/Shape_1Shapegru_42/transpose:y:0*
T0*
_output_shapes
:2
gru_42/Shape_1?
gru_42/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_42/strided_slice_1/stack?
gru_42/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_42/strided_slice_1/stack_1?
gru_42/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_42/strided_slice_1/stack_2?
gru_42/strided_slice_1StridedSlicegru_42/Shape_1:output:0%gru_42/strided_slice_1/stack:output:0'gru_42/strided_slice_1/stack_1:output:0'gru_42/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_42/strided_slice_1?
"gru_42/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_42/TensorArrayV2/element_shape?
gru_42/TensorArrayV2TensorListReserve+gru_42/TensorArrayV2/element_shape:output:0gru_42/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_42/TensorArrayV2?
<gru_42/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_42/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_42/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_42/transpose:y:0Egru_42/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_42/TensorArrayUnstack/TensorListFromTensor?
gru_42/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_42/strided_slice_2/stack?
gru_42/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_42/strided_slice_2/stack_1?
gru_42/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_42/strided_slice_2/stack_2?
gru_42/strided_slice_2StridedSlicegru_42/transpose:y:0%gru_42/strided_slice_2/stack:output:0'gru_42/strided_slice_2/stack_1:output:0'gru_42/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_42/strided_slice_2?
!gru_42/gru_cell_42/ReadVariableOpReadVariableOp*gru_42_gru_cell_42_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_42/gru_cell_42/ReadVariableOp?
gru_42/gru_cell_42/unstackUnpack)gru_42/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_42/gru_cell_42/unstack?
(gru_42/gru_cell_42/MatMul/ReadVariableOpReadVariableOp1gru_42_gru_cell_42_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_42/gru_cell_42/MatMul/ReadVariableOp?
gru_42/gru_cell_42/MatMulMatMulgru_42/strided_slice_2:output:00gru_42/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/MatMul?
gru_42/gru_cell_42/BiasAddBiasAdd#gru_42/gru_cell_42/MatMul:product:0#gru_42/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/BiasAdd?
"gru_42/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_42/gru_cell_42/split/split_dim?
gru_42/gru_cell_42/splitSplit+gru_42/gru_cell_42/split/split_dim:output:0#gru_42/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_42/gru_cell_42/split?
*gru_42/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp3gru_42_gru_cell_42_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_42/gru_cell_42/MatMul_1/ReadVariableOp?
gru_42/gru_cell_42/MatMul_1MatMulgru_42/zeros:output:02gru_42/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/MatMul_1?
gru_42/gru_cell_42/BiasAdd_1BiasAdd%gru_42/gru_cell_42/MatMul_1:product:0#gru_42/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/BiasAdd_1?
gru_42/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_42/gru_cell_42/Const?
$gru_42/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_42/gru_cell_42/split_1/split_dim?
gru_42/gru_cell_42/split_1SplitV%gru_42/gru_cell_42/BiasAdd_1:output:0!gru_42/gru_cell_42/Const:output:0-gru_42/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_42/gru_cell_42/split_1?
gru_42/gru_cell_42/addAddV2!gru_42/gru_cell_42/split:output:0#gru_42/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/add?
gru_42/gru_cell_42/SigmoidSigmoidgru_42/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/Sigmoid?
gru_42/gru_cell_42/add_1AddV2!gru_42/gru_cell_42/split:output:1#gru_42/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/add_1?
gru_42/gru_cell_42/Sigmoid_1Sigmoidgru_42/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/Sigmoid_1?
gru_42/gru_cell_42/mulMul gru_42/gru_cell_42/Sigmoid_1:y:0#gru_42/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/mul?
gru_42/gru_cell_42/add_2AddV2!gru_42/gru_cell_42/split:output:2gru_42/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/add_2?
gru_42/gru_cell_42/ReluRelugru_42/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/Relu?
gru_42/gru_cell_42/mul_1Mulgru_42/gru_cell_42/Sigmoid:y:0gru_42/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/mul_1y
gru_42/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_42/gru_cell_42/sub/x?
gru_42/gru_cell_42/subSub!gru_42/gru_cell_42/sub/x:output:0gru_42/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/sub?
gru_42/gru_cell_42/mul_2Mulgru_42/gru_cell_42/sub:z:0%gru_42/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/mul_2?
gru_42/gru_cell_42/add_3AddV2gru_42/gru_cell_42/mul_1:z:0gru_42/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/add_3?
$gru_42/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2&
$gru_42/TensorArrayV2_1/element_shape?
gru_42/TensorArrayV2_1TensorListReserve-gru_42/TensorArrayV2_1/element_shape:output:0gru_42/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_42/TensorArrayV2_1\
gru_42/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_42/time?
gru_42/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_42/while/maximum_iterationsx
gru_42/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_42/while/loop_counter?
gru_42/whileWhile"gru_42/while/loop_counter:output:0(gru_42/while/maximum_iterations:output:0gru_42/time:output:0gru_42/TensorArrayV2_1:handle:0gru_42/zeros:output:0gru_42/strided_slice_1:output:0>gru_42/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_42_gru_cell_42_readvariableop_resource1gru_42_gru_cell_42_matmul_readvariableop_resource3gru_42_gru_cell_42_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_42_while_body_1574698*%
condR
gru_42_while_cond_1574697*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_42/while?
7gru_42/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7gru_42/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_42/TensorArrayV2Stack/TensorListStackTensorListStackgru_42/while:output:3@gru_42/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_42/TensorArrayV2Stack/TensorListStack?
gru_42/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_42/strided_slice_3/stack?
gru_42/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_42/strided_slice_3/stack_1?
gru_42/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_42/strided_slice_3/stack_2?
gru_42/strided_slice_3StridedSlice2gru_42/TensorArrayV2Stack/TensorListStack:tensor:0%gru_42/strided_slice_3/stack:output:0'gru_42/strided_slice_3/stack_1:output:0'gru_42/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_42/strided_slice_3?
gru_42/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_42/transpose_1/perm?
gru_42/transpose_1	Transpose2gru_42/TensorArrayV2Stack/TensorListStack:tensor:0 gru_42/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_42/transpose_1t
gru_42/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_42/runtimey
dropout_63/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_63/dropout/Const?
dropout_63/dropout/MulMulgru_42/transpose_1:y:0!dropout_63/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_63/dropout/Mulz
dropout_63/dropout/ShapeShapegru_42/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_63/dropout/Shape?
/dropout_63/dropout/random_uniform/RandomUniformRandomUniform!dropout_63/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_63/dropout/random_uniform/RandomUniform?
!dropout_63/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_63/dropout/GreaterEqual/y?
dropout_63/dropout/GreaterEqualGreaterEqual8dropout_63/dropout/random_uniform/RandomUniform:output:0*dropout_63/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_63/dropout/GreaterEqual?
dropout_63/dropout/CastCast#dropout_63/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_63/dropout/Cast?
dropout_63/dropout/Mul_1Muldropout_63/dropout/Mul:z:0dropout_63/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_63/dropout/Mul_1h
gru_43/ShapeShapedropout_63/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
gru_43/Shape?
gru_43/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_43/strided_slice/stack?
gru_43/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_43/strided_slice/stack_1?
gru_43/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_43/strided_slice/stack_2?
gru_43/strided_sliceStridedSlicegru_43/Shape:output:0#gru_43/strided_slice/stack:output:0%gru_43/strided_slice/stack_1:output:0%gru_43/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_43/strided_sliceq
gru_43/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_43/zeros/packed/1?
gru_43/zeros/packedPackgru_43/strided_slice:output:0gru_43/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_43/zeros/packedm
gru_43/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_43/zeros/Const?
gru_43/zerosFillgru_43/zeros/packed:output:0gru_43/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_43/zeros?
gru_43/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_43/transpose/perm?
gru_43/transpose	Transposedropout_63/dropout/Mul_1:z:0gru_43/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_43/transposed
gru_43/Shape_1Shapegru_43/transpose:y:0*
T0*
_output_shapes
:2
gru_43/Shape_1?
gru_43/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_43/strided_slice_1/stack?
gru_43/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_43/strided_slice_1/stack_1?
gru_43/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_43/strided_slice_1/stack_2?
gru_43/strided_slice_1StridedSlicegru_43/Shape_1:output:0%gru_43/strided_slice_1/stack:output:0'gru_43/strided_slice_1/stack_1:output:0'gru_43/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_43/strided_slice_1?
"gru_43/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_43/TensorArrayV2/element_shape?
gru_43/TensorArrayV2TensorListReserve+gru_43/TensorArrayV2/element_shape:output:0gru_43/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_43/TensorArrayV2?
<gru_43/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2>
<gru_43/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_43/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_43/transpose:y:0Egru_43/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_43/TensorArrayUnstack/TensorListFromTensor?
gru_43/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_43/strided_slice_2/stack?
gru_43/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_43/strided_slice_2/stack_1?
gru_43/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_43/strided_slice_2/stack_2?
gru_43/strided_slice_2StridedSlicegru_43/transpose:y:0%gru_43/strided_slice_2/stack:output:0'gru_43/strided_slice_2/stack_1:output:0'gru_43/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_43/strided_slice_2?
!gru_43/gru_cell_43/ReadVariableOpReadVariableOp*gru_43_gru_cell_43_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_43/gru_cell_43/ReadVariableOp?
gru_43/gru_cell_43/unstackUnpack)gru_43/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_43/gru_cell_43/unstack?
(gru_43/gru_cell_43/MatMul/ReadVariableOpReadVariableOp1gru_43_gru_cell_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(gru_43/gru_cell_43/MatMul/ReadVariableOp?
gru_43/gru_cell_43/MatMulMatMulgru_43/strided_slice_2:output:00gru_43/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/MatMul?
gru_43/gru_cell_43/BiasAddBiasAdd#gru_43/gru_cell_43/MatMul:product:0#gru_43/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/BiasAdd?
"gru_43/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_43/gru_cell_43/split/split_dim?
gru_43/gru_cell_43/splitSplit+gru_43/gru_cell_43/split/split_dim:output:0#gru_43/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_43/gru_cell_43/split?
*gru_43/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp3gru_43_gru_cell_43_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_43/gru_cell_43/MatMul_1/ReadVariableOp?
gru_43/gru_cell_43/MatMul_1MatMulgru_43/zeros:output:02gru_43/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/MatMul_1?
gru_43/gru_cell_43/BiasAdd_1BiasAdd%gru_43/gru_cell_43/MatMul_1:product:0#gru_43/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/BiasAdd_1?
gru_43/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_43/gru_cell_43/Const?
$gru_43/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_43/gru_cell_43/split_1/split_dim?
gru_43/gru_cell_43/split_1SplitV%gru_43/gru_cell_43/BiasAdd_1:output:0!gru_43/gru_cell_43/Const:output:0-gru_43/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_43/gru_cell_43/split_1?
gru_43/gru_cell_43/addAddV2!gru_43/gru_cell_43/split:output:0#gru_43/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/add?
gru_43/gru_cell_43/SigmoidSigmoidgru_43/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/Sigmoid?
gru_43/gru_cell_43/add_1AddV2!gru_43/gru_cell_43/split:output:1#gru_43/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/add_1?
gru_43/gru_cell_43/Sigmoid_1Sigmoidgru_43/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/Sigmoid_1?
gru_43/gru_cell_43/mulMul gru_43/gru_cell_43/Sigmoid_1:y:0#gru_43/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/mul?
gru_43/gru_cell_43/add_2AddV2!gru_43/gru_cell_43/split:output:2gru_43/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/add_2?
gru_43/gru_cell_43/ReluRelugru_43/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/Relu?
gru_43/gru_cell_43/mul_1Mulgru_43/gru_cell_43/Sigmoid:y:0gru_43/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/mul_1y
gru_43/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_43/gru_cell_43/sub/x?
gru_43/gru_cell_43/subSub!gru_43/gru_cell_43/sub/x:output:0gru_43/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/sub?
gru_43/gru_cell_43/mul_2Mulgru_43/gru_cell_43/sub:z:0%gru_43/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/mul_2?
gru_43/gru_cell_43/add_3AddV2gru_43/gru_cell_43/mul_1:z:0gru_43/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/add_3?
$gru_43/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2&
$gru_43/TensorArrayV2_1/element_shape?
gru_43/TensorArrayV2_1TensorListReserve-gru_43/TensorArrayV2_1/element_shape:output:0gru_43/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_43/TensorArrayV2_1\
gru_43/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_43/time?
gru_43/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_43/while/maximum_iterationsx
gru_43/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_43/while/loop_counter?
gru_43/whileWhile"gru_43/while/loop_counter:output:0(gru_43/while/maximum_iterations:output:0gru_43/time:output:0gru_43/TensorArrayV2_1:handle:0gru_43/zeros:output:0gru_43/strided_slice_1:output:0>gru_43/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_43_gru_cell_43_readvariableop_resource1gru_43_gru_cell_43_matmul_readvariableop_resource3gru_43_gru_cell_43_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_43_while_body_1574855*%
condR
gru_43_while_cond_1574854*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_43/while?
7gru_43/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7gru_43/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_43/TensorArrayV2Stack/TensorListStackTensorListStackgru_43/while:output:3@gru_43/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_43/TensorArrayV2Stack/TensorListStack?
gru_43/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_43/strided_slice_3/stack?
gru_43/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_43/strided_slice_3/stack_1?
gru_43/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_43/strided_slice_3/stack_2?
gru_43/strided_slice_3StridedSlice2gru_43/TensorArrayV2Stack/TensorListStack:tensor:0%gru_43/strided_slice_3/stack:output:0'gru_43/strided_slice_3/stack_1:output:0'gru_43/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_43/strided_slice_3?
gru_43/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_43/transpose_1/perm?
gru_43/transpose_1	Transpose2gru_43/TensorArrayV2Stack/TensorListStack:tensor:0 gru_43/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_43/transpose_1t
gru_43/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_43/runtimey
dropout_64/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_64/dropout/Const?
dropout_64/dropout/MulMulgru_43/transpose_1:y:0!dropout_64/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_64/dropout/Mulz
dropout_64/dropout/ShapeShapegru_43/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_64/dropout/Shape?
/dropout_64/dropout/random_uniform/RandomUniformRandomUniform!dropout_64/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_64/dropout/random_uniform/RandomUniform?
!dropout_64/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_64/dropout/GreaterEqual/y?
dropout_64/dropout/GreaterEqualGreaterEqual8dropout_64/dropout/random_uniform/RandomUniform:output:0*dropout_64/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_64/dropout/GreaterEqual?
dropout_64/dropout/CastCast#dropout_64/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_64/dropout/Cast?
dropout_64/dropout/Mul_1Muldropout_64/dropout/Mul:z:0dropout_64/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_64/dropout/Mul_1?
!dense_42/Tensordot/ReadVariableOpReadVariableOp*dense_42_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_42/Tensordot/ReadVariableOp|
dense_42/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_42/Tensordot/axes?
dense_42/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_42/Tensordot/free?
dense_42/Tensordot/ShapeShapedropout_64/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_42/Tensordot/Shape?
 dense_42/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_42/Tensordot/GatherV2/axis?
dense_42/Tensordot/GatherV2GatherV2!dense_42/Tensordot/Shape:output:0 dense_42/Tensordot/free:output:0)dense_42/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_42/Tensordot/GatherV2?
"dense_42/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_42/Tensordot/GatherV2_1/axis?
dense_42/Tensordot/GatherV2_1GatherV2!dense_42/Tensordot/Shape:output:0 dense_42/Tensordot/axes:output:0+dense_42/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_42/Tensordot/GatherV2_1~
dense_42/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_42/Tensordot/Const?
dense_42/Tensordot/ProdProd$dense_42/Tensordot/GatherV2:output:0!dense_42/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_42/Tensordot/Prod?
dense_42/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_42/Tensordot/Const_1?
dense_42/Tensordot/Prod_1Prod&dense_42/Tensordot/GatherV2_1:output:0#dense_42/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_42/Tensordot/Prod_1?
dense_42/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_42/Tensordot/concat/axis?
dense_42/Tensordot/concatConcatV2 dense_42/Tensordot/free:output:0 dense_42/Tensordot/axes:output:0'dense_42/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_42/Tensordot/concat?
dense_42/Tensordot/stackPack dense_42/Tensordot/Prod:output:0"dense_42/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_42/Tensordot/stack?
dense_42/Tensordot/transpose	Transposedropout_64/dropout/Mul_1:z:0"dense_42/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_42/Tensordot/transpose?
dense_42/Tensordot/ReshapeReshape dense_42/Tensordot/transpose:y:0!dense_42/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_42/Tensordot/Reshape?
dense_42/Tensordot/MatMulMatMul#dense_42/Tensordot/Reshape:output:0)dense_42/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_42/Tensordot/MatMul?
dense_42/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_42/Tensordot/Const_2?
 dense_42/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_42/Tensordot/concat_1/axis?
dense_42/Tensordot/concat_1ConcatV2$dense_42/Tensordot/GatherV2:output:0#dense_42/Tensordot/Const_2:output:0)dense_42/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_42/Tensordot/concat_1?
dense_42/TensordotReshape#dense_42/Tensordot/MatMul:product:0$dense_42/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_42/Tensordot?
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_42/BiasAdd/ReadVariableOp?
dense_42/BiasAddBiasAdddense_42/Tensordot:output:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_42/BiasAddx
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_42/Reluy
dropout_65/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_65/dropout/Const?
dropout_65/dropout/MulMuldense_42/Relu:activations:0!dropout_65/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_65/dropout/Mul
dropout_65/dropout/ShapeShapedense_42/Relu:activations:0*
T0*
_output_shapes
:2
dropout_65/dropout/Shape?
/dropout_65/dropout/random_uniform/RandomUniformRandomUniform!dropout_65/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_65/dropout/random_uniform/RandomUniform?
!dropout_65/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_65/dropout/GreaterEqual/y?
dropout_65/dropout/GreaterEqualGreaterEqual8dropout_65/dropout/random_uniform/RandomUniform:output:0*dropout_65/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_65/dropout/GreaterEqual?
dropout_65/dropout/CastCast#dropout_65/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_65/dropout/Cast?
dropout_65/dropout/Mul_1Muldropout_65/dropout/Mul:z:0dropout_65/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_65/dropout/Mul_1?
!dense_43/Tensordot/ReadVariableOpReadVariableOp*dense_43_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_43/Tensordot/ReadVariableOp|
dense_43/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_43/Tensordot/axes?
dense_43/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_43/Tensordot/free?
dense_43/Tensordot/ShapeShapedropout_65/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_43/Tensordot/Shape?
 dense_43/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_43/Tensordot/GatherV2/axis?
dense_43/Tensordot/GatherV2GatherV2!dense_43/Tensordot/Shape:output:0 dense_43/Tensordot/free:output:0)dense_43/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_43/Tensordot/GatherV2?
"dense_43/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_43/Tensordot/GatherV2_1/axis?
dense_43/Tensordot/GatherV2_1GatherV2!dense_43/Tensordot/Shape:output:0 dense_43/Tensordot/axes:output:0+dense_43/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_43/Tensordot/GatherV2_1~
dense_43/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_43/Tensordot/Const?
dense_43/Tensordot/ProdProd$dense_43/Tensordot/GatherV2:output:0!dense_43/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_43/Tensordot/Prod?
dense_43/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_43/Tensordot/Const_1?
dense_43/Tensordot/Prod_1Prod&dense_43/Tensordot/GatherV2_1:output:0#dense_43/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_43/Tensordot/Prod_1?
dense_43/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_43/Tensordot/concat/axis?
dense_43/Tensordot/concatConcatV2 dense_43/Tensordot/free:output:0 dense_43/Tensordot/axes:output:0'dense_43/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_43/Tensordot/concat?
dense_43/Tensordot/stackPack dense_43/Tensordot/Prod:output:0"dense_43/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_43/Tensordot/stack?
dense_43/Tensordot/transpose	Transposedropout_65/dropout/Mul_1:z:0"dense_43/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_43/Tensordot/transpose?
dense_43/Tensordot/ReshapeReshape dense_43/Tensordot/transpose:y:0!dense_43/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_43/Tensordot/Reshape?
dense_43/Tensordot/MatMulMatMul#dense_43/Tensordot/Reshape:output:0)dense_43/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_43/Tensordot/MatMul?
dense_43/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_43/Tensordot/Const_2?
 dense_43/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_43/Tensordot/concat_1/axis?
dense_43/Tensordot/concat_1ConcatV2$dense_43/Tensordot/GatherV2:output:0#dense_43/Tensordot/Const_2:output:0)dense_43/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_43/Tensordot/concat_1?
dense_43/TensordotReshape#dense_43/Tensordot/MatMul:product:0$dense_43/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_43/Tensordot?
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_43/BiasAdd/ReadVariableOp?
dense_43/BiasAddBiasAdddense_43/Tensordot:output:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_43/BiasAddx
dense_43/ReluReludense_43/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_43/Reluy
dropout_66/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_66/dropout/Const?
dropout_66/dropout/MulMuldense_43/Relu:activations:0!dropout_66/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_66/dropout/Mul
dropout_66/dropout/ShapeShapedense_43/Relu:activations:0*
T0*
_output_shapes
:2
dropout_66/dropout/Shape?
/dropout_66/dropout/random_uniform/RandomUniformRandomUniform!dropout_66/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_66/dropout/random_uniform/RandomUniform?
!dropout_66/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_66/dropout/GreaterEqual/y?
dropout_66/dropout/GreaterEqualGreaterEqual8dropout_66/dropout/random_uniform/RandomUniform:output:0*dropout_66/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_66/dropout/GreaterEqual?
dropout_66/dropout/CastCast#dropout_66/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_66/dropout/Cast?
dropout_66/dropout/Mul_1Muldropout_66/dropout/Mul:z:0dropout_66/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_66/dropout/Mul_1?
!dense_44/Tensordot/ReadVariableOpReadVariableOp*dense_44_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_44/Tensordot/ReadVariableOp|
dense_44/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_44/Tensordot/axes?
dense_44/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_44/Tensordot/free?
dense_44/Tensordot/ShapeShapedropout_66/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_44/Tensordot/Shape?
 dense_44/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_44/Tensordot/GatherV2/axis?
dense_44/Tensordot/GatherV2GatherV2!dense_44/Tensordot/Shape:output:0 dense_44/Tensordot/free:output:0)dense_44/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_44/Tensordot/GatherV2?
"dense_44/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_44/Tensordot/GatherV2_1/axis?
dense_44/Tensordot/GatherV2_1GatherV2!dense_44/Tensordot/Shape:output:0 dense_44/Tensordot/axes:output:0+dense_44/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_44/Tensordot/GatherV2_1~
dense_44/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_44/Tensordot/Const?
dense_44/Tensordot/ProdProd$dense_44/Tensordot/GatherV2:output:0!dense_44/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_44/Tensordot/Prod?
dense_44/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_44/Tensordot/Const_1?
dense_44/Tensordot/Prod_1Prod&dense_44/Tensordot/GatherV2_1:output:0#dense_44/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_44/Tensordot/Prod_1?
dense_44/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_44/Tensordot/concat/axis?
dense_44/Tensordot/concatConcatV2 dense_44/Tensordot/free:output:0 dense_44/Tensordot/axes:output:0'dense_44/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_44/Tensordot/concat?
dense_44/Tensordot/stackPack dense_44/Tensordot/Prod:output:0"dense_44/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_44/Tensordot/stack?
dense_44/Tensordot/transpose	Transposedropout_66/dropout/Mul_1:z:0"dense_44/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_44/Tensordot/transpose?
dense_44/Tensordot/ReshapeReshape dense_44/Tensordot/transpose:y:0!dense_44/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_44/Tensordot/Reshape?
dense_44/Tensordot/MatMulMatMul#dense_44/Tensordot/Reshape:output:0)dense_44/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_44/Tensordot/MatMul?
dense_44/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_44/Tensordot/Const_2?
 dense_44/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_44/Tensordot/concat_1/axis?
dense_44/Tensordot/concat_1ConcatV2$dense_44/Tensordot/GatherV2:output:0#dense_44/Tensordot/Const_2:output:0)dense_44/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_44/Tensordot/concat_1?
dense_44/TensordotReshape#dense_44/Tensordot/MatMul:product:0$dense_44/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_44/Tensordot?
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOp?
dense_44/BiasAddBiasAdddense_44/Tensordot:output:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_44/BiasAddx
IdentityIdentitydense_44/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_42/BiasAdd/ReadVariableOp"^dense_42/Tensordot/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp"^dense_43/Tensordot/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp"^dense_44/Tensordot/ReadVariableOp)^gru_42/gru_cell_42/MatMul/ReadVariableOp+^gru_42/gru_cell_42/MatMul_1/ReadVariableOp"^gru_42/gru_cell_42/ReadVariableOp^gru_42/while)^gru_43/gru_cell_43/MatMul/ReadVariableOp+^gru_43/gru_cell_43/MatMul_1/ReadVariableOp"^gru_43/gru_cell_43/ReadVariableOp^gru_43/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2F
!dense_42/Tensordot/ReadVariableOp!dense_42/Tensordot/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2F
!dense_43/Tensordot/ReadVariableOp!dense_43/Tensordot/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2F
!dense_44/Tensordot/ReadVariableOp!dense_44/Tensordot/ReadVariableOp2T
(gru_42/gru_cell_42/MatMul/ReadVariableOp(gru_42/gru_cell_42/MatMul/ReadVariableOp2X
*gru_42/gru_cell_42/MatMul_1/ReadVariableOp*gru_42/gru_cell_42/MatMul_1/ReadVariableOp2F
!gru_42/gru_cell_42/ReadVariableOp!gru_42/gru_cell_42/ReadVariableOp2
gru_42/whilegru_42/while2T
(gru_43/gru_cell_43/MatMul/ReadVariableOp(gru_43/gru_cell_43/MatMul/ReadVariableOp2X
*gru_43/gru_cell_43/MatMul_1/ReadVariableOp*gru_43/gru_cell_43/MatMul_1/ReadVariableOp2F
!gru_43/gru_cell_43/ReadVariableOp!gru_43/gru_cell_43/ReadVariableOp2
gru_43/whilegru_43/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
,__inference_dropout_63_layer_call_fn_1575789

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_63_layer_call_and_return_conditional_losses_15738352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?E
?
while_body_1576312
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_43_readvariableop_resource_0:	?F
2while_gru_cell_43_matmul_readvariableop_resource_0:
??H
4while_gru_cell_43_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_43_readvariableop_resource:	?D
0while_gru_cell_43_matmul_readvariableop_resource:
??F
2while_gru_cell_43_matmul_1_readvariableop_resource:
????'while/gru_cell_43/MatMul/ReadVariableOp?)while/gru_cell_43/MatMul_1/ReadVariableOp? while/gru_cell_43/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_43/ReadVariableOpReadVariableOp+while_gru_cell_43_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_43/ReadVariableOp?
while/gru_cell_43/unstackUnpack(while/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_43/unstack?
'while/gru_cell_43/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_43_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_43/MatMul/ReadVariableOp?
while/gru_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul?
while/gru_cell_43/BiasAddBiasAdd"while/gru_cell_43/MatMul:product:0"while/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd?
!while/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_43/split/split_dim?
while/gru_cell_43/splitSplit*while/gru_cell_43/split/split_dim:output:0"while/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split?
)while/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_43_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_43/MatMul_1/ReadVariableOp?
while/gru_cell_43/MatMul_1MatMulwhile_placeholder_21while/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul_1?
while/gru_cell_43/BiasAdd_1BiasAdd$while/gru_cell_43/MatMul_1:product:0"while/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd_1?
while/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_43/Const?
#while/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_43/split_1/split_dim?
while/gru_cell_43/split_1SplitV$while/gru_cell_43/BiasAdd_1:output:0 while/gru_cell_43/Const:output:0,while/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split_1?
while/gru_cell_43/addAddV2 while/gru_cell_43/split:output:0"while/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add?
while/gru_cell_43/SigmoidSigmoidwhile/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid?
while/gru_cell_43/add_1AddV2 while/gru_cell_43/split:output:1"while/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_1?
while/gru_cell_43/Sigmoid_1Sigmoidwhile/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid_1?
while/gru_cell_43/mulMulwhile/gru_cell_43/Sigmoid_1:y:0"while/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul?
while/gru_cell_43/add_2AddV2 while/gru_cell_43/split:output:2while/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_2?
while/gru_cell_43/ReluReluwhile/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Relu?
while/gru_cell_43/mul_1Mulwhile/gru_cell_43/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_1w
while/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_43/sub/x?
while/gru_cell_43/subSub while/gru_cell_43/sub/x:output:0while/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/sub?
while/gru_cell_43/mul_2Mulwhile/gru_cell_43/sub:z:0$while/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_2?
while/gru_cell_43/add_3AddV2while/gru_cell_43/mul_1:z:0while/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_43/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_43/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_43/MatMul/ReadVariableOp*^while/gru_cell_43/MatMul_1/ReadVariableOp!^while/gru_cell_43/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_43_matmul_1_readvariableop_resource4while_gru_cell_43_matmul_1_readvariableop_resource_0"f
0while_gru_cell_43_matmul_readvariableop_resource2while_gru_cell_43_matmul_readvariableop_resource_0"X
)while_gru_cell_43_readvariableop_resource+while_gru_cell_43_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_43/MatMul/ReadVariableOp'while/gru_cell_43/MatMul/ReadVariableOp2V
)while/gru_cell_43/MatMul_1/ReadVariableOp)while/gru_cell_43/MatMul_1/ReadVariableOp2D
 while/gru_cell_43/ReadVariableOp while/gru_cell_43/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?;
?
C__inference_gru_42_layer_call_and_return_conditional_losses_1572256

inputs&
gru_cell_42_1572180:	?&
gru_cell_42_1572182:	?'
gru_cell_42_1572184:
??
identity??#gru_cell_42/StatefulPartitionedCall?whileD
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
B :?2
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
:??????????2
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
 :??????????????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_42_1572180gru_cell_42_1572182gru_cell_42_1572184*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_15721292%
#gru_cell_42/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_42_1572180gru_cell_42_1572182gru_cell_42_1572184*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1572192*
condR
while_cond_1572191*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identity|
NoOpNoOp$^gru_cell_42/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2J
#gru_cell_42/StatefulPartitionedCall#gru_cell_42/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_1575852
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1575852___redundant_placeholder05
1while_while_cond_1575852___redundant_placeholder15
1while_while_cond_1575852___redundant_placeholder25
1while_while_cond_1575852___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_1572695

inputs

states*
readvariableop_resource:	?2
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
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
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
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
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
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
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

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
.:??????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
while_cond_1575628
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1575628___redundant_placeholder05
1while_while_cond_1575628___redundant_placeholder15
1while_while_cond_1575628___redundant_placeholder25
1while_while_cond_1575628___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_1572129

inputs

states*
readvariableop_resource:	?1
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
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
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
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
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
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
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

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
-:?????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
*__inference_dense_42_layer_call_fn_1576512

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_15734202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
-__inference_gru_cell_43_layer_call_fn_1576843

inputs
states_0
unknown:	?
	unknown_0:
??
	unknown_1:
??
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
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_15725522
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

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
.:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?

?
-__inference_gru_cell_42_layer_call_fn_1576751

inputs
states_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
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
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_15721292
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

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
-:?????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
(__inference_gru_43_layer_call_fn_1576445

inputs
unknown:	?
	unknown_0:
??
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_43_layer_call_and_return_conditional_losses_15738062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?f
?
'sequential_21_gru_42_while_body_1571594F
Bsequential_21_gru_42_while_sequential_21_gru_42_while_loop_counterL
Hsequential_21_gru_42_while_sequential_21_gru_42_while_maximum_iterations*
&sequential_21_gru_42_while_placeholder,
(sequential_21_gru_42_while_placeholder_1,
(sequential_21_gru_42_while_placeholder_2E
Asequential_21_gru_42_while_sequential_21_gru_42_strided_slice_1_0?
}sequential_21_gru_42_while_tensorarrayv2read_tensorlistgetitem_sequential_21_gru_42_tensorarrayunstack_tensorlistfromtensor_0S
@sequential_21_gru_42_while_gru_cell_42_readvariableop_resource_0:	?Z
Gsequential_21_gru_42_while_gru_cell_42_matmul_readvariableop_resource_0:	?]
Isequential_21_gru_42_while_gru_cell_42_matmul_1_readvariableop_resource_0:
??'
#sequential_21_gru_42_while_identity)
%sequential_21_gru_42_while_identity_1)
%sequential_21_gru_42_while_identity_2)
%sequential_21_gru_42_while_identity_3)
%sequential_21_gru_42_while_identity_4C
?sequential_21_gru_42_while_sequential_21_gru_42_strided_slice_1
{sequential_21_gru_42_while_tensorarrayv2read_tensorlistgetitem_sequential_21_gru_42_tensorarrayunstack_tensorlistfromtensorQ
>sequential_21_gru_42_while_gru_cell_42_readvariableop_resource:	?X
Esequential_21_gru_42_while_gru_cell_42_matmul_readvariableop_resource:	?[
Gsequential_21_gru_42_while_gru_cell_42_matmul_1_readvariableop_resource:
????<sequential_21/gru_42/while/gru_cell_42/MatMul/ReadVariableOp?>sequential_21/gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp?5sequential_21/gru_42/while/gru_cell_42/ReadVariableOp?
Lsequential_21/gru_42/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2N
Lsequential_21/gru_42/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_21/gru_42/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_21_gru_42_while_tensorarrayv2read_tensorlistgetitem_sequential_21_gru_42_tensorarrayunstack_tensorlistfromtensor_0&sequential_21_gru_42_while_placeholderUsequential_21/gru_42/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02@
>sequential_21/gru_42/while/TensorArrayV2Read/TensorListGetItem?
5sequential_21/gru_42/while/gru_cell_42/ReadVariableOpReadVariableOp@sequential_21_gru_42_while_gru_cell_42_readvariableop_resource_0*
_output_shapes
:	?*
dtype027
5sequential_21/gru_42/while/gru_cell_42/ReadVariableOp?
.sequential_21/gru_42/while/gru_cell_42/unstackUnpack=sequential_21/gru_42/while/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num20
.sequential_21/gru_42/while/gru_cell_42/unstack?
<sequential_21/gru_42/while/gru_cell_42/MatMul/ReadVariableOpReadVariableOpGsequential_21_gru_42_while_gru_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02>
<sequential_21/gru_42/while/gru_cell_42/MatMul/ReadVariableOp?
-sequential_21/gru_42/while/gru_cell_42/MatMulMatMulEsequential_21/gru_42/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential_21/gru_42/while/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_21/gru_42/while/gru_cell_42/MatMul?
.sequential_21/gru_42/while/gru_cell_42/BiasAddBiasAdd7sequential_21/gru_42/while/gru_cell_42/MatMul:product:07sequential_21/gru_42/while/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????20
.sequential_21/gru_42/while/gru_cell_42/BiasAdd?
6sequential_21/gru_42/while/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_21/gru_42/while/gru_cell_42/split/split_dim?
,sequential_21/gru_42/while/gru_cell_42/splitSplit?sequential_21/gru_42/while/gru_cell_42/split/split_dim:output:07sequential_21/gru_42/while/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2.
,sequential_21/gru_42/while/gru_cell_42/split?
>sequential_21/gru_42/while/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOpIsequential_21_gru_42_while_gru_cell_42_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02@
>sequential_21/gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp?
/sequential_21/gru_42/while/gru_cell_42/MatMul_1MatMul(sequential_21_gru_42_while_placeholder_2Fsequential_21/gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_21/gru_42/while/gru_cell_42/MatMul_1?
0sequential_21/gru_42/while/gru_cell_42/BiasAdd_1BiasAdd9sequential_21/gru_42/while/gru_cell_42/MatMul_1:product:07sequential_21/gru_42/while/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????22
0sequential_21/gru_42/while/gru_cell_42/BiasAdd_1?
,sequential_21/gru_42/while/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2.
,sequential_21/gru_42/while/gru_cell_42/Const?
8sequential_21/gru_42/while/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8sequential_21/gru_42/while/gru_cell_42/split_1/split_dim?
.sequential_21/gru_42/while/gru_cell_42/split_1SplitV9sequential_21/gru_42/while/gru_cell_42/BiasAdd_1:output:05sequential_21/gru_42/while/gru_cell_42/Const:output:0Asequential_21/gru_42/while/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split20
.sequential_21/gru_42/while/gru_cell_42/split_1?
*sequential_21/gru_42/while/gru_cell_42/addAddV25sequential_21/gru_42/while/gru_cell_42/split:output:07sequential_21/gru_42/while/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2,
*sequential_21/gru_42/while/gru_cell_42/add?
.sequential_21/gru_42/while/gru_cell_42/SigmoidSigmoid.sequential_21/gru_42/while/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????20
.sequential_21/gru_42/while/gru_cell_42/Sigmoid?
,sequential_21/gru_42/while/gru_cell_42/add_1AddV25sequential_21/gru_42/while/gru_cell_42/split:output:17sequential_21/gru_42/while/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2.
,sequential_21/gru_42/while/gru_cell_42/add_1?
0sequential_21/gru_42/while/gru_cell_42/Sigmoid_1Sigmoid0sequential_21/gru_42/while/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????22
0sequential_21/gru_42/while/gru_cell_42/Sigmoid_1?
*sequential_21/gru_42/while/gru_cell_42/mulMul4sequential_21/gru_42/while/gru_cell_42/Sigmoid_1:y:07sequential_21/gru_42/while/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2,
*sequential_21/gru_42/while/gru_cell_42/mul?
,sequential_21/gru_42/while/gru_cell_42/add_2AddV25sequential_21/gru_42/while/gru_cell_42/split:output:2.sequential_21/gru_42/while/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2.
,sequential_21/gru_42/while/gru_cell_42/add_2?
+sequential_21/gru_42/while/gru_cell_42/ReluRelu0sequential_21/gru_42/while/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2-
+sequential_21/gru_42/while/gru_cell_42/Relu?
,sequential_21/gru_42/while/gru_cell_42/mul_1Mul2sequential_21/gru_42/while/gru_cell_42/Sigmoid:y:0(sequential_21_gru_42_while_placeholder_2*
T0*(
_output_shapes
:??????????2.
,sequential_21/gru_42/while/gru_cell_42/mul_1?
,sequential_21/gru_42/while/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,sequential_21/gru_42/while/gru_cell_42/sub/x?
*sequential_21/gru_42/while/gru_cell_42/subSub5sequential_21/gru_42/while/gru_cell_42/sub/x:output:02sequential_21/gru_42/while/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2,
*sequential_21/gru_42/while/gru_cell_42/sub?
,sequential_21/gru_42/while/gru_cell_42/mul_2Mul.sequential_21/gru_42/while/gru_cell_42/sub:z:09sequential_21/gru_42/while/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,sequential_21/gru_42/while/gru_cell_42/mul_2?
,sequential_21/gru_42/while/gru_cell_42/add_3AddV20sequential_21/gru_42/while/gru_cell_42/mul_1:z:00sequential_21/gru_42/while/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2.
,sequential_21/gru_42/while/gru_cell_42/add_3?
?sequential_21/gru_42/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_21_gru_42_while_placeholder_1&sequential_21_gru_42_while_placeholder0sequential_21/gru_42/while/gru_cell_42/add_3:z:0*
_output_shapes
: *
element_dtype02A
?sequential_21/gru_42/while/TensorArrayV2Write/TensorListSetItem?
 sequential_21/gru_42/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_21/gru_42/while/add/y?
sequential_21/gru_42/while/addAddV2&sequential_21_gru_42_while_placeholder)sequential_21/gru_42/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_21/gru_42/while/add?
"sequential_21/gru_42/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_21/gru_42/while/add_1/y?
 sequential_21/gru_42/while/add_1AddV2Bsequential_21_gru_42_while_sequential_21_gru_42_while_loop_counter+sequential_21/gru_42/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_21/gru_42/while/add_1?
#sequential_21/gru_42/while/IdentityIdentity$sequential_21/gru_42/while/add_1:z:0 ^sequential_21/gru_42/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_21/gru_42/while/Identity?
%sequential_21/gru_42/while/Identity_1IdentityHsequential_21_gru_42_while_sequential_21_gru_42_while_maximum_iterations ^sequential_21/gru_42/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_21/gru_42/while/Identity_1?
%sequential_21/gru_42/while/Identity_2Identity"sequential_21/gru_42/while/add:z:0 ^sequential_21/gru_42/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_21/gru_42/while/Identity_2?
%sequential_21/gru_42/while/Identity_3IdentityOsequential_21/gru_42/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_21/gru_42/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_21/gru_42/while/Identity_3?
%sequential_21/gru_42/while/Identity_4Identity0sequential_21/gru_42/while/gru_cell_42/add_3:z:0 ^sequential_21/gru_42/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_21/gru_42/while/Identity_4?
sequential_21/gru_42/while/NoOpNoOp=^sequential_21/gru_42/while/gru_cell_42/MatMul/ReadVariableOp?^sequential_21/gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp6^sequential_21/gru_42/while/gru_cell_42/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_21/gru_42/while/NoOp"?
Gsequential_21_gru_42_while_gru_cell_42_matmul_1_readvariableop_resourceIsequential_21_gru_42_while_gru_cell_42_matmul_1_readvariableop_resource_0"?
Esequential_21_gru_42_while_gru_cell_42_matmul_readvariableop_resourceGsequential_21_gru_42_while_gru_cell_42_matmul_readvariableop_resource_0"?
>sequential_21_gru_42_while_gru_cell_42_readvariableop_resource@sequential_21_gru_42_while_gru_cell_42_readvariableop_resource_0"S
#sequential_21_gru_42_while_identity,sequential_21/gru_42/while/Identity:output:0"W
%sequential_21_gru_42_while_identity_1.sequential_21/gru_42/while/Identity_1:output:0"W
%sequential_21_gru_42_while_identity_2.sequential_21/gru_42/while/Identity_2:output:0"W
%sequential_21_gru_42_while_identity_3.sequential_21/gru_42/while/Identity_3:output:0"W
%sequential_21_gru_42_while_identity_4.sequential_21/gru_42/while/Identity_4:output:0"?
?sequential_21_gru_42_while_sequential_21_gru_42_strided_slice_1Asequential_21_gru_42_while_sequential_21_gru_42_strided_slice_1_0"?
{sequential_21_gru_42_while_tensorarrayv2read_tensorlistgetitem_sequential_21_gru_42_tensorarrayunstack_tensorlistfromtensor}sequential_21_gru_42_while_tensorarrayv2read_tensorlistgetitem_sequential_21_gru_42_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2|
<sequential_21/gru_42/while/gru_cell_42/MatMul/ReadVariableOp<sequential_21/gru_42/while/gru_cell_42/MatMul/ReadVariableOp2?
>sequential_21/gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp>sequential_21/gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp2n
5sequential_21/gru_42/while/gru_cell_42/ReadVariableOp5sequential_21/gru_42/while/gru_cell_42/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?
e
,__inference_dropout_66_layer_call_fn_1576606

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_66_layer_call_and_return_conditional_losses_15735712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?Y
?
C__inference_gru_42_layer_call_and_return_conditional_losses_1575259
inputs_06
#gru_cell_42_readvariableop_resource:	?=
*gru_cell_42_matmul_readvariableop_resource:	?@
,gru_cell_42_matmul_1_readvariableop_resource:
??
identity??!gru_cell_42/MatMul/ReadVariableOp?#gru_cell_42/MatMul_1/ReadVariableOp?gru_cell_42/ReadVariableOp?whileF
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
B :?2
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
:??????????2
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
 :??????????????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_42/ReadVariableOpReadVariableOp#gru_cell_42_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_42/ReadVariableOp?
gru_cell_42/unstackUnpack"gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_42/unstack?
!gru_cell_42/MatMul/ReadVariableOpReadVariableOp*gru_cell_42_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_42/MatMul/ReadVariableOp?
gru_cell_42/MatMulMatMulstrided_slice_2:output:0)gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul?
gru_cell_42/BiasAddBiasAddgru_cell_42/MatMul:product:0gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd?
gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split/split_dim?
gru_cell_42/splitSplit$gru_cell_42/split/split_dim:output:0gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split?
#gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_42_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_42/MatMul_1/ReadVariableOp?
gru_cell_42/MatMul_1MatMulzeros:output:0+gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul_1?
gru_cell_42/BiasAdd_1BiasAddgru_cell_42/MatMul_1:product:0gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd_1{
gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_42/Const?
gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split_1/split_dim?
gru_cell_42/split_1SplitVgru_cell_42/BiasAdd_1:output:0gru_cell_42/Const:output:0&gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split_1?
gru_cell_42/addAddV2gru_cell_42/split:output:0gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add}
gru_cell_42/SigmoidSigmoidgru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid?
gru_cell_42/add_1AddV2gru_cell_42/split:output:1gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_1?
gru_cell_42/Sigmoid_1Sigmoidgru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid_1?
gru_cell_42/mulMulgru_cell_42/Sigmoid_1:y:0gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul?
gru_cell_42/add_2AddV2gru_cell_42/split:output:2gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_2v
gru_cell_42/ReluRelugru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Relu?
gru_cell_42/mul_1Mulgru_cell_42/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_1k
gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_42/sub/x?
gru_cell_42/subSubgru_cell_42/sub/x:output:0gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/sub?
gru_cell_42/mul_2Mulgru_cell_42/sub:z:0gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_2?
gru_cell_42/add_3AddV2gru_cell_42/mul_1:z:0gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_42_readvariableop_resource*gru_cell_42_matmul_readvariableop_resource,gru_cell_42_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1575170*
condR
while_cond_1575169*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identity?
NoOpNoOp"^gru_cell_42/MatMul/ReadVariableOp$^gru_cell_42/MatMul_1/ReadVariableOp^gru_cell_42/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2F
!gru_cell_42/MatMul/ReadVariableOp!gru_cell_42/MatMul/ReadVariableOp2J
#gru_cell_42/MatMul_1/ReadVariableOp#gru_cell_42/MatMul_1/ReadVariableOp28
gru_cell_42/ReadVariableOpgru_cell_42/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
e
G__inference_dropout_66_layer_call_and_return_conditional_losses_1573475

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
E__inference_dense_42_layer_call_and_return_conditional_losses_1576503

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_66_layer_call_fn_1576601

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_66_layer_call_and_return_conditional_losses_15734752
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_21_layer_call_fn_1573541
gru_42_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_42_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_15735142
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
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_42_input
?
e
G__inference_dropout_65_layer_call_and_return_conditional_losses_1576517

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_1575169
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1575169___redundant_placeholder05
1while_while_cond_1575169___redundant_placeholder15
1while_while_cond_1575169___redundant_placeholder25
1while_while_cond_1575169___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?E
?
while_body_1575476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_42_readvariableop_resource_0:	?E
2while_gru_cell_42_matmul_readvariableop_resource_0:	?H
4while_gru_cell_42_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_42_readvariableop_resource:	?C
0while_gru_cell_42_matmul_readvariableop_resource:	?F
2while_gru_cell_42_matmul_1_readvariableop_resource:
????'while/gru_cell_42/MatMul/ReadVariableOp?)while/gru_cell_42/MatMul_1/ReadVariableOp? while/gru_cell_42/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_42/ReadVariableOpReadVariableOp+while_gru_cell_42_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_42/ReadVariableOp?
while/gru_cell_42/unstackUnpack(while/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_42/unstack?
'while/gru_cell_42/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_42/MatMul/ReadVariableOp?
while/gru_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul?
while/gru_cell_42/BiasAddBiasAdd"while/gru_cell_42/MatMul:product:0"while/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd?
!while/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_42/split/split_dim?
while/gru_cell_42/splitSplit*while/gru_cell_42/split/split_dim:output:0"while/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split?
)while/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_42_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_42/MatMul_1/ReadVariableOp?
while/gru_cell_42/MatMul_1MatMulwhile_placeholder_21while/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul_1?
while/gru_cell_42/BiasAdd_1BiasAdd$while/gru_cell_42/MatMul_1:product:0"while/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd_1?
while/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_42/Const?
#while/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_42/split_1/split_dim?
while/gru_cell_42/split_1SplitV$while/gru_cell_42/BiasAdd_1:output:0 while/gru_cell_42/Const:output:0,while/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split_1?
while/gru_cell_42/addAddV2 while/gru_cell_42/split:output:0"while/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add?
while/gru_cell_42/SigmoidSigmoidwhile/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid?
while/gru_cell_42/add_1AddV2 while/gru_cell_42/split:output:1"while/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_1?
while/gru_cell_42/Sigmoid_1Sigmoidwhile/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid_1?
while/gru_cell_42/mulMulwhile/gru_cell_42/Sigmoid_1:y:0"while/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul?
while/gru_cell_42/add_2AddV2 while/gru_cell_42/split:output:2while/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_2?
while/gru_cell_42/ReluReluwhile/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Relu?
while/gru_cell_42/mul_1Mulwhile/gru_cell_42/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_1w
while/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_42/sub/x?
while/gru_cell_42/subSub while/gru_cell_42/sub/x:output:0while/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/sub?
while/gru_cell_42/mul_2Mulwhile/gru_cell_42/sub:z:0$while/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_2?
while/gru_cell_42/add_3AddV2while/gru_cell_42/mul_1:z:0while/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_42/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_42/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_42/MatMul/ReadVariableOp*^while/gru_cell_42/MatMul_1/ReadVariableOp!^while/gru_cell_42/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_42_matmul_1_readvariableop_resource4while_gru_cell_42_matmul_1_readvariableop_resource_0"f
0while_gru_cell_42_matmul_readvariableop_resource2while_gru_cell_42_matmul_readvariableop_resource_0"X
)while_gru_cell_42_readvariableop_resource+while_gru_cell_42_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_42/MatMul/ReadVariableOp'while/gru_cell_42/MatMul/ReadVariableOp2V
)while/gru_cell_42/MatMul_1/ReadVariableOp)while/gru_cell_42/MatMul_1/ReadVariableOp2D
 while/gru_cell_42/ReadVariableOp while/gru_cell_42/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?
f
G__inference_dropout_63_layer_call_and_return_conditional_losses_1573835

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?Y
?
C__inference_gru_42_layer_call_and_return_conditional_losses_1575412
inputs_06
#gru_cell_42_readvariableop_resource:	?=
*gru_cell_42_matmul_readvariableop_resource:	?@
,gru_cell_42_matmul_1_readvariableop_resource:
??
identity??!gru_cell_42/MatMul/ReadVariableOp?#gru_cell_42/MatMul_1/ReadVariableOp?gru_cell_42/ReadVariableOp?whileF
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
B :?2
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
:??????????2
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
 :??????????????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_42/ReadVariableOpReadVariableOp#gru_cell_42_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_42/ReadVariableOp?
gru_cell_42/unstackUnpack"gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_42/unstack?
!gru_cell_42/MatMul/ReadVariableOpReadVariableOp*gru_cell_42_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_42/MatMul/ReadVariableOp?
gru_cell_42/MatMulMatMulstrided_slice_2:output:0)gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul?
gru_cell_42/BiasAddBiasAddgru_cell_42/MatMul:product:0gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd?
gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split/split_dim?
gru_cell_42/splitSplit$gru_cell_42/split/split_dim:output:0gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split?
#gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_42_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_42/MatMul_1/ReadVariableOp?
gru_cell_42/MatMul_1MatMulzeros:output:0+gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul_1?
gru_cell_42/BiasAdd_1BiasAddgru_cell_42/MatMul_1:product:0gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd_1{
gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_42/Const?
gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split_1/split_dim?
gru_cell_42/split_1SplitVgru_cell_42/BiasAdd_1:output:0gru_cell_42/Const:output:0&gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split_1?
gru_cell_42/addAddV2gru_cell_42/split:output:0gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add}
gru_cell_42/SigmoidSigmoidgru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid?
gru_cell_42/add_1AddV2gru_cell_42/split:output:1gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_1?
gru_cell_42/Sigmoid_1Sigmoidgru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid_1?
gru_cell_42/mulMulgru_cell_42/Sigmoid_1:y:0gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul?
gru_cell_42/add_2AddV2gru_cell_42/split:output:2gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_2v
gru_cell_42/ReluRelugru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Relu?
gru_cell_42/mul_1Mulgru_cell_42/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_1k
gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_42/sub/x?
gru_cell_42/subSubgru_cell_42/sub/x:output:0gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/sub?
gru_cell_42/mul_2Mulgru_cell_42/sub:z:0gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_2?
gru_cell_42/add_3AddV2gru_cell_42/mul_1:z:0gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_42_readvariableop_resource*gru_cell_42_matmul_readvariableop_resource,gru_cell_42_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1575323*
condR
while_cond_1575322*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identity?
NoOpNoOp"^gru_cell_42/MatMul/ReadVariableOp$^gru_cell_42/MatMul_1/ReadVariableOp^gru_cell_42/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2F
!gru_cell_42/MatMul/ReadVariableOp!gru_cell_42/MatMul/ReadVariableOp2J
#gru_cell_42/MatMul_1/ReadVariableOp#gru_cell_42/MatMul_1/ReadVariableOp28
gru_cell_42/ReadVariableOpgru_cell_42/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
(__inference_gru_43_layer_call_fn_1576412
inputs_0
unknown:	?
	unknown_0:
??
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_43_layer_call_and_return_conditional_losses_15726292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_1572191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1572191___redundant_placeholder05
1while_while_cond_1572191___redundant_placeholder15
1while_while_cond_1572191___redundant_placeholder25
1while_while_cond_1572191___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
-__inference_gru_cell_42_layer_call_fn_1576737

inputs
states_0
unknown:	?
	unknown_0:	?
	unknown_1:
??
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
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_15719862
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

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
-:?????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
(__inference_gru_42_layer_call_fn_1575751

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_42_layer_call_and_return_conditional_losses_15732072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_1577160
file_prefix4
 assignvariableop_dense_42_kernel:
??/
 assignvariableop_1_dense_42_bias:	?6
"assignvariableop_2_dense_43_kernel:
??/
 assignvariableop_3_dense_43_bias:	?5
"assignvariableop_4_dense_44_kernel:	?.
 assignvariableop_5_dense_44_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: @
-assignvariableop_11_gru_42_gru_cell_42_kernel:	?K
7assignvariableop_12_gru_42_gru_cell_42_recurrent_kernel:
??>
+assignvariableop_13_gru_42_gru_cell_42_bias:	?A
-assignvariableop_14_gru_43_gru_cell_43_kernel:
??K
7assignvariableop_15_gru_43_gru_cell_43_recurrent_kernel:
??>
+assignvariableop_16_gru_43_gru_cell_43_bias:	?#
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: >
*assignvariableop_21_adam_dense_42_kernel_m:
??7
(assignvariableop_22_adam_dense_42_bias_m:	?>
*assignvariableop_23_adam_dense_43_kernel_m:
??7
(assignvariableop_24_adam_dense_43_bias_m:	?=
*assignvariableop_25_adam_dense_44_kernel_m:	?6
(assignvariableop_26_adam_dense_44_bias_m:G
4assignvariableop_27_adam_gru_42_gru_cell_42_kernel_m:	?R
>assignvariableop_28_adam_gru_42_gru_cell_42_recurrent_kernel_m:
??E
2assignvariableop_29_adam_gru_42_gru_cell_42_bias_m:	?H
4assignvariableop_30_adam_gru_43_gru_cell_43_kernel_m:
??R
>assignvariableop_31_adam_gru_43_gru_cell_43_recurrent_kernel_m:
??E
2assignvariableop_32_adam_gru_43_gru_cell_43_bias_m:	?>
*assignvariableop_33_adam_dense_42_kernel_v:
??7
(assignvariableop_34_adam_dense_42_bias_v:	?>
*assignvariableop_35_adam_dense_43_kernel_v:
??7
(assignvariableop_36_adam_dense_43_bias_v:	?=
*assignvariableop_37_adam_dense_44_kernel_v:	?6
(assignvariableop_38_adam_dense_44_bias_v:G
4assignvariableop_39_adam_gru_42_gru_cell_42_kernel_v:	?R
>assignvariableop_40_adam_gru_42_gru_cell_42_recurrent_kernel_v:
??E
2assignvariableop_41_adam_gru_42_gru_cell_42_bias_v:	?H
4assignvariableop_42_adam_gru_43_gru_cell_43_kernel_v:
??R
>assignvariableop_43_adam_gru_43_gru_cell_43_recurrent_kernel_v:
??E
2assignvariableop_44_adam_gru_43_gru_cell_43_bias_v:	?
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp assignvariableop_dense_42_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_42_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_43_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_43_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_44_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_44_biasIdentity_5:output:0"/device:CPU:0*
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
AssignVariableOp_11AssignVariableOp-assignvariableop_11_gru_42_gru_cell_42_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_gru_42_gru_cell_42_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_gru_42_gru_cell_42_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp-assignvariableop_14_gru_43_gru_cell_43_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp7assignvariableop_15_gru_43_gru_cell_43_recurrent_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_gru_43_gru_cell_43_biasIdentity_16:output:0"/device:CPU:0*
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
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_42_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_42_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_43_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_43_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_44_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_44_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_gru_42_gru_cell_42_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_gru_42_gru_cell_42_recurrent_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_gru_42_gru_cell_42_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_gru_43_gru_cell_43_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_gru_43_gru_cell_43_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_gru_43_gru_cell_43_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_42_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_42_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_43_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_43_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_44_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_44_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_gru_42_gru_cell_42_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp>assignvariableop_40_adam_gru_42_gru_cell_42_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_gru_42_gru_cell_42_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adam_gru_43_gru_cell_43_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp>assignvariableop_43_adam_gru_43_gru_cell_43_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp2assignvariableop_44_adam_gru_43_gru_cell_43_bias_vIdentity_44:output:0"/device:CPU:0*
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
?
?
while_cond_1571998
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1571998___redundant_placeholder05
1while_while_cond_1571998___redundant_placeholder15
1while_while_cond_1571998___redundant_placeholder25
1while_while_cond_1571998___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1576311
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1576311___redundant_placeholder05
1while_while_cond_1576311___redundant_placeholder15
1while_while_cond_1576311___redundant_placeholder25
1while_while_cond_1576311___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?E
?
while_body_1575853
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_43_readvariableop_resource_0:	?F
2while_gru_cell_43_matmul_readvariableop_resource_0:
??H
4while_gru_cell_43_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_43_readvariableop_resource:	?D
0while_gru_cell_43_matmul_readvariableop_resource:
??F
2while_gru_cell_43_matmul_1_readvariableop_resource:
????'while/gru_cell_43/MatMul/ReadVariableOp?)while/gru_cell_43/MatMul_1/ReadVariableOp? while/gru_cell_43/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_43/ReadVariableOpReadVariableOp+while_gru_cell_43_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_43/ReadVariableOp?
while/gru_cell_43/unstackUnpack(while/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_43/unstack?
'while/gru_cell_43/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_43_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_43/MatMul/ReadVariableOp?
while/gru_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul?
while/gru_cell_43/BiasAddBiasAdd"while/gru_cell_43/MatMul:product:0"while/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd?
!while/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_43/split/split_dim?
while/gru_cell_43/splitSplit*while/gru_cell_43/split/split_dim:output:0"while/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split?
)while/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_43_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_43/MatMul_1/ReadVariableOp?
while/gru_cell_43/MatMul_1MatMulwhile_placeholder_21while/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul_1?
while/gru_cell_43/BiasAdd_1BiasAdd$while/gru_cell_43/MatMul_1:product:0"while/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd_1?
while/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_43/Const?
#while/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_43/split_1/split_dim?
while/gru_cell_43/split_1SplitV$while/gru_cell_43/BiasAdd_1:output:0 while/gru_cell_43/Const:output:0,while/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split_1?
while/gru_cell_43/addAddV2 while/gru_cell_43/split:output:0"while/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add?
while/gru_cell_43/SigmoidSigmoidwhile/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid?
while/gru_cell_43/add_1AddV2 while/gru_cell_43/split:output:1"while/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_1?
while/gru_cell_43/Sigmoid_1Sigmoidwhile/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid_1?
while/gru_cell_43/mulMulwhile/gru_cell_43/Sigmoid_1:y:0"while/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul?
while/gru_cell_43/add_2AddV2 while/gru_cell_43/split:output:2while/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_2?
while/gru_cell_43/ReluReluwhile/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Relu?
while/gru_cell_43/mul_1Mulwhile/gru_cell_43/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_1w
while/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_43/sub/x?
while/gru_cell_43/subSub while/gru_cell_43/sub/x:output:0while/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/sub?
while/gru_cell_43/mul_2Mulwhile/gru_cell_43/sub:z:0$while/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_2?
while/gru_cell_43/add_3AddV2while/gru_cell_43/mul_1:z:0while/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_43/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_43/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_43/MatMul/ReadVariableOp*^while/gru_cell_43/MatMul_1/ReadVariableOp!^while/gru_cell_43/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_43_matmul_1_readvariableop_resource4while_gru_cell_43_matmul_1_readvariableop_resource_0"f
0while_gru_cell_43_matmul_readvariableop_resource2while_gru_cell_43_matmul_readvariableop_resource_0"X
)while_gru_cell_43_readvariableop_resource+while_gru_cell_43_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_43/MatMul/ReadVariableOp'while/gru_cell_43/MatMul/ReadVariableOp2V
)while/gru_cell_43/MatMul_1/ReadVariableOp)while/gru_cell_43/MatMul_1/ReadVariableOp2D
 while/gru_cell_43/ReadVariableOp while/gru_cell_43/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?
H
,__inference_dropout_65_layer_call_fn_1576534

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_65_layer_call_and_return_conditional_losses_15734312
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?X
?
C__inference_gru_42_layer_call_and_return_conditional_losses_1575718

inputs6
#gru_cell_42_readvariableop_resource:	?=
*gru_cell_42_matmul_readvariableop_resource:	?@
,gru_cell_42_matmul_1_readvariableop_resource:
??
identity??!gru_cell_42/MatMul/ReadVariableOp?#gru_cell_42/MatMul_1/ReadVariableOp?gru_cell_42/ReadVariableOp?whileD
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
B :?2
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
:??????????2
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
:?????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_42/ReadVariableOpReadVariableOp#gru_cell_42_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_42/ReadVariableOp?
gru_cell_42/unstackUnpack"gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_42/unstack?
!gru_cell_42/MatMul/ReadVariableOpReadVariableOp*gru_cell_42_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_42/MatMul/ReadVariableOp?
gru_cell_42/MatMulMatMulstrided_slice_2:output:0)gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul?
gru_cell_42/BiasAddBiasAddgru_cell_42/MatMul:product:0gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd?
gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split/split_dim?
gru_cell_42/splitSplit$gru_cell_42/split/split_dim:output:0gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split?
#gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_42_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_42/MatMul_1/ReadVariableOp?
gru_cell_42/MatMul_1MatMulzeros:output:0+gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul_1?
gru_cell_42/BiasAdd_1BiasAddgru_cell_42/MatMul_1:product:0gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd_1{
gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_42/Const?
gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split_1/split_dim?
gru_cell_42/split_1SplitVgru_cell_42/BiasAdd_1:output:0gru_cell_42/Const:output:0&gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split_1?
gru_cell_42/addAddV2gru_cell_42/split:output:0gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add}
gru_cell_42/SigmoidSigmoidgru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid?
gru_cell_42/add_1AddV2gru_cell_42/split:output:1gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_1?
gru_cell_42/Sigmoid_1Sigmoidgru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid_1?
gru_cell_42/mulMulgru_cell_42/Sigmoid_1:y:0gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul?
gru_cell_42/add_2AddV2gru_cell_42/split:output:2gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_2v
gru_cell_42/ReluRelugru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Relu?
gru_cell_42/mul_1Mulgru_cell_42/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_1k
gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_42/sub/x?
gru_cell_42/subSubgru_cell_42/sub/x:output:0gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/sub?
gru_cell_42/mul_2Mulgru_cell_42/sub:z:0gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_2?
gru_cell_42/add_3AddV2gru_cell_42/mul_1:z:0gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_42_readvariableop_resource*gru_cell_42_matmul_readvariableop_resource,gru_cell_42_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1575629*
condR
while_cond_1575628*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
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
:??????????2
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
:??????????2

Identity?
NoOpNoOp"^gru_cell_42/MatMul/ReadVariableOp$^gru_cell_42/MatMul_1/ReadVariableOp^gru_cell_42/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_42/MatMul/ReadVariableOp!gru_cell_42/MatMul/ReadVariableOp2J
#gru_cell_42/MatMul_1/ReadVariableOp#gru_cell_42/MatMul_1/ReadVariableOp28
gru_cell_42/ReadVariableOpgru_cell_42/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_64_layer_call_and_return_conditional_losses_1573637

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?

J__inference_sequential_21_layer_call_and_return_conditional_losses_1574634

inputs=
*gru_42_gru_cell_42_readvariableop_resource:	?D
1gru_42_gru_cell_42_matmul_readvariableop_resource:	?G
3gru_42_gru_cell_42_matmul_1_readvariableop_resource:
??=
*gru_43_gru_cell_43_readvariableop_resource:	?E
1gru_43_gru_cell_43_matmul_readvariableop_resource:
??G
3gru_43_gru_cell_43_matmul_1_readvariableop_resource:
??>
*dense_42_tensordot_readvariableop_resource:
??7
(dense_42_biasadd_readvariableop_resource:	?>
*dense_43_tensordot_readvariableop_resource:
??7
(dense_43_biasadd_readvariableop_resource:	?=
*dense_44_tensordot_readvariableop_resource:	?6
(dense_44_biasadd_readvariableop_resource:
identity??dense_42/BiasAdd/ReadVariableOp?!dense_42/Tensordot/ReadVariableOp?dense_43/BiasAdd/ReadVariableOp?!dense_43/Tensordot/ReadVariableOp?dense_44/BiasAdd/ReadVariableOp?!dense_44/Tensordot/ReadVariableOp?(gru_42/gru_cell_42/MatMul/ReadVariableOp?*gru_42/gru_cell_42/MatMul_1/ReadVariableOp?!gru_42/gru_cell_42/ReadVariableOp?gru_42/while?(gru_43/gru_cell_43/MatMul/ReadVariableOp?*gru_43/gru_cell_43/MatMul_1/ReadVariableOp?!gru_43/gru_cell_43/ReadVariableOp?gru_43/whileR
gru_42/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_42/Shape?
gru_42/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_42/strided_slice/stack?
gru_42/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_42/strided_slice/stack_1?
gru_42/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_42/strided_slice/stack_2?
gru_42/strided_sliceStridedSlicegru_42/Shape:output:0#gru_42/strided_slice/stack:output:0%gru_42/strided_slice/stack_1:output:0%gru_42/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_42/strided_sliceq
gru_42/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_42/zeros/packed/1?
gru_42/zeros/packedPackgru_42/strided_slice:output:0gru_42/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_42/zeros/packedm
gru_42/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_42/zeros/Const?
gru_42/zerosFillgru_42/zeros/packed:output:0gru_42/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_42/zeros?
gru_42/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_42/transpose/perm?
gru_42/transpose	Transposeinputsgru_42/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
gru_42/transposed
gru_42/Shape_1Shapegru_42/transpose:y:0*
T0*
_output_shapes
:2
gru_42/Shape_1?
gru_42/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_42/strided_slice_1/stack?
gru_42/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_42/strided_slice_1/stack_1?
gru_42/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_42/strided_slice_1/stack_2?
gru_42/strided_slice_1StridedSlicegru_42/Shape_1:output:0%gru_42/strided_slice_1/stack:output:0'gru_42/strided_slice_1/stack_1:output:0'gru_42/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_42/strided_slice_1?
"gru_42/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_42/TensorArrayV2/element_shape?
gru_42/TensorArrayV2TensorListReserve+gru_42/TensorArrayV2/element_shape:output:0gru_42/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_42/TensorArrayV2?
<gru_42/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_42/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_42/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_42/transpose:y:0Egru_42/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_42/TensorArrayUnstack/TensorListFromTensor?
gru_42/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_42/strided_slice_2/stack?
gru_42/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_42/strided_slice_2/stack_1?
gru_42/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_42/strided_slice_2/stack_2?
gru_42/strided_slice_2StridedSlicegru_42/transpose:y:0%gru_42/strided_slice_2/stack:output:0'gru_42/strided_slice_2/stack_1:output:0'gru_42/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_42/strided_slice_2?
!gru_42/gru_cell_42/ReadVariableOpReadVariableOp*gru_42_gru_cell_42_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_42/gru_cell_42/ReadVariableOp?
gru_42/gru_cell_42/unstackUnpack)gru_42/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_42/gru_cell_42/unstack?
(gru_42/gru_cell_42/MatMul/ReadVariableOpReadVariableOp1gru_42_gru_cell_42_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_42/gru_cell_42/MatMul/ReadVariableOp?
gru_42/gru_cell_42/MatMulMatMulgru_42/strided_slice_2:output:00gru_42/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/MatMul?
gru_42/gru_cell_42/BiasAddBiasAdd#gru_42/gru_cell_42/MatMul:product:0#gru_42/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/BiasAdd?
"gru_42/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_42/gru_cell_42/split/split_dim?
gru_42/gru_cell_42/splitSplit+gru_42/gru_cell_42/split/split_dim:output:0#gru_42/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_42/gru_cell_42/split?
*gru_42/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp3gru_42_gru_cell_42_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_42/gru_cell_42/MatMul_1/ReadVariableOp?
gru_42/gru_cell_42/MatMul_1MatMulgru_42/zeros:output:02gru_42/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/MatMul_1?
gru_42/gru_cell_42/BiasAdd_1BiasAdd%gru_42/gru_cell_42/MatMul_1:product:0#gru_42/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/BiasAdd_1?
gru_42/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_42/gru_cell_42/Const?
$gru_42/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_42/gru_cell_42/split_1/split_dim?
gru_42/gru_cell_42/split_1SplitV%gru_42/gru_cell_42/BiasAdd_1:output:0!gru_42/gru_cell_42/Const:output:0-gru_42/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_42/gru_cell_42/split_1?
gru_42/gru_cell_42/addAddV2!gru_42/gru_cell_42/split:output:0#gru_42/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/add?
gru_42/gru_cell_42/SigmoidSigmoidgru_42/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/Sigmoid?
gru_42/gru_cell_42/add_1AddV2!gru_42/gru_cell_42/split:output:1#gru_42/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/add_1?
gru_42/gru_cell_42/Sigmoid_1Sigmoidgru_42/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/Sigmoid_1?
gru_42/gru_cell_42/mulMul gru_42/gru_cell_42/Sigmoid_1:y:0#gru_42/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/mul?
gru_42/gru_cell_42/add_2AddV2!gru_42/gru_cell_42/split:output:2gru_42/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/add_2?
gru_42/gru_cell_42/ReluRelugru_42/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/Relu?
gru_42/gru_cell_42/mul_1Mulgru_42/gru_cell_42/Sigmoid:y:0gru_42/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/mul_1y
gru_42/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_42/gru_cell_42/sub/x?
gru_42/gru_cell_42/subSub!gru_42/gru_cell_42/sub/x:output:0gru_42/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/sub?
gru_42/gru_cell_42/mul_2Mulgru_42/gru_cell_42/sub:z:0%gru_42/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/mul_2?
gru_42/gru_cell_42/add_3AddV2gru_42/gru_cell_42/mul_1:z:0gru_42/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_42/gru_cell_42/add_3?
$gru_42/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2&
$gru_42/TensorArrayV2_1/element_shape?
gru_42/TensorArrayV2_1TensorListReserve-gru_42/TensorArrayV2_1/element_shape:output:0gru_42/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_42/TensorArrayV2_1\
gru_42/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_42/time?
gru_42/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_42/while/maximum_iterationsx
gru_42/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_42/while/loop_counter?
gru_42/whileWhile"gru_42/while/loop_counter:output:0(gru_42/while/maximum_iterations:output:0gru_42/time:output:0gru_42/TensorArrayV2_1:handle:0gru_42/zeros:output:0gru_42/strided_slice_1:output:0>gru_42/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_42_gru_cell_42_readvariableop_resource1gru_42_gru_cell_42_matmul_readvariableop_resource3gru_42_gru_cell_42_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_42_while_body_1574312*%
condR
gru_42_while_cond_1574311*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_42/while?
7gru_42/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7gru_42/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_42/TensorArrayV2Stack/TensorListStackTensorListStackgru_42/while:output:3@gru_42/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_42/TensorArrayV2Stack/TensorListStack?
gru_42/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_42/strided_slice_3/stack?
gru_42/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_42/strided_slice_3/stack_1?
gru_42/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_42/strided_slice_3/stack_2?
gru_42/strided_slice_3StridedSlice2gru_42/TensorArrayV2Stack/TensorListStack:tensor:0%gru_42/strided_slice_3/stack:output:0'gru_42/strided_slice_3/stack_1:output:0'gru_42/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_42/strided_slice_3?
gru_42/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_42/transpose_1/perm?
gru_42/transpose_1	Transpose2gru_42/TensorArrayV2Stack/TensorListStack:tensor:0 gru_42/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_42/transpose_1t
gru_42/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_42/runtime?
dropout_63/IdentityIdentitygru_42/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_63/Identityh
gru_43/ShapeShapedropout_63/Identity:output:0*
T0*
_output_shapes
:2
gru_43/Shape?
gru_43/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_43/strided_slice/stack?
gru_43/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_43/strided_slice/stack_1?
gru_43/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_43/strided_slice/stack_2?
gru_43/strided_sliceStridedSlicegru_43/Shape:output:0#gru_43/strided_slice/stack:output:0%gru_43/strided_slice/stack_1:output:0%gru_43/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_43/strided_sliceq
gru_43/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
gru_43/zeros/packed/1?
gru_43/zeros/packedPackgru_43/strided_slice:output:0gru_43/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_43/zeros/packedm
gru_43/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_43/zeros/Const?
gru_43/zerosFillgru_43/zeros/packed:output:0gru_43/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_43/zeros?
gru_43/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_43/transpose/perm?
gru_43/transpose	Transposedropout_63/Identity:output:0gru_43/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_43/transposed
gru_43/Shape_1Shapegru_43/transpose:y:0*
T0*
_output_shapes
:2
gru_43/Shape_1?
gru_43/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_43/strided_slice_1/stack?
gru_43/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_43/strided_slice_1/stack_1?
gru_43/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_43/strided_slice_1/stack_2?
gru_43/strided_slice_1StridedSlicegru_43/Shape_1:output:0%gru_43/strided_slice_1/stack:output:0'gru_43/strided_slice_1/stack_1:output:0'gru_43/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_43/strided_slice_1?
"gru_43/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_43/TensorArrayV2/element_shape?
gru_43/TensorArrayV2TensorListReserve+gru_43/TensorArrayV2/element_shape:output:0gru_43/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_43/TensorArrayV2?
<gru_43/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2>
<gru_43/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_43/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_43/transpose:y:0Egru_43/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_43/TensorArrayUnstack/TensorListFromTensor?
gru_43/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_43/strided_slice_2/stack?
gru_43/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_43/strided_slice_2/stack_1?
gru_43/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_43/strided_slice_2/stack_2?
gru_43/strided_slice_2StridedSlicegru_43/transpose:y:0%gru_43/strided_slice_2/stack:output:0'gru_43/strided_slice_2/stack_1:output:0'gru_43/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_43/strided_slice_2?
!gru_43/gru_cell_43/ReadVariableOpReadVariableOp*gru_43_gru_cell_43_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_43/gru_cell_43/ReadVariableOp?
gru_43/gru_cell_43/unstackUnpack)gru_43/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_43/gru_cell_43/unstack?
(gru_43/gru_cell_43/MatMul/ReadVariableOpReadVariableOp1gru_43_gru_cell_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(gru_43/gru_cell_43/MatMul/ReadVariableOp?
gru_43/gru_cell_43/MatMulMatMulgru_43/strided_slice_2:output:00gru_43/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/MatMul?
gru_43/gru_cell_43/BiasAddBiasAdd#gru_43/gru_cell_43/MatMul:product:0#gru_43/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/BiasAdd?
"gru_43/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_43/gru_cell_43/split/split_dim?
gru_43/gru_cell_43/splitSplit+gru_43/gru_cell_43/split/split_dim:output:0#gru_43/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_43/gru_cell_43/split?
*gru_43/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp3gru_43_gru_cell_43_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*gru_43/gru_cell_43/MatMul_1/ReadVariableOp?
gru_43/gru_cell_43/MatMul_1MatMulgru_43/zeros:output:02gru_43/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/MatMul_1?
gru_43/gru_cell_43/BiasAdd_1BiasAdd%gru_43/gru_cell_43/MatMul_1:product:0#gru_43/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/BiasAdd_1?
gru_43/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_43/gru_cell_43/Const?
$gru_43/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_43/gru_cell_43/split_1/split_dim?
gru_43/gru_cell_43/split_1SplitV%gru_43/gru_cell_43/BiasAdd_1:output:0!gru_43/gru_cell_43/Const:output:0-gru_43/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_43/gru_cell_43/split_1?
gru_43/gru_cell_43/addAddV2!gru_43/gru_cell_43/split:output:0#gru_43/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/add?
gru_43/gru_cell_43/SigmoidSigmoidgru_43/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/Sigmoid?
gru_43/gru_cell_43/add_1AddV2!gru_43/gru_cell_43/split:output:1#gru_43/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/add_1?
gru_43/gru_cell_43/Sigmoid_1Sigmoidgru_43/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/Sigmoid_1?
gru_43/gru_cell_43/mulMul gru_43/gru_cell_43/Sigmoid_1:y:0#gru_43/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/mul?
gru_43/gru_cell_43/add_2AddV2!gru_43/gru_cell_43/split:output:2gru_43/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/add_2?
gru_43/gru_cell_43/ReluRelugru_43/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/Relu?
gru_43/gru_cell_43/mul_1Mulgru_43/gru_cell_43/Sigmoid:y:0gru_43/zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/mul_1y
gru_43/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_43/gru_cell_43/sub/x?
gru_43/gru_cell_43/subSub!gru_43/gru_cell_43/sub/x:output:0gru_43/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/sub?
gru_43/gru_cell_43/mul_2Mulgru_43/gru_cell_43/sub:z:0%gru_43/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/mul_2?
gru_43/gru_cell_43/add_3AddV2gru_43/gru_cell_43/mul_1:z:0gru_43/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_43/gru_cell_43/add_3?
$gru_43/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2&
$gru_43/TensorArrayV2_1/element_shape?
gru_43/TensorArrayV2_1TensorListReserve-gru_43/TensorArrayV2_1/element_shape:output:0gru_43/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_43/TensorArrayV2_1\
gru_43/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_43/time?
gru_43/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_43/while/maximum_iterationsx
gru_43/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_43/while/loop_counter?
gru_43/whileWhile"gru_43/while/loop_counter:output:0(gru_43/while/maximum_iterations:output:0gru_43/time:output:0gru_43/TensorArrayV2_1:handle:0gru_43/zeros:output:0gru_43/strided_slice_1:output:0>gru_43/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_43_gru_cell_43_readvariableop_resource1gru_43_gru_cell_43_matmul_readvariableop_resource3gru_43_gru_cell_43_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_43_while_body_1574462*%
condR
gru_43_while_cond_1574461*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
gru_43/while?
7gru_43/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7gru_43/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_43/TensorArrayV2Stack/TensorListStackTensorListStackgru_43/while:output:3@gru_43/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02+
)gru_43/TensorArrayV2Stack/TensorListStack?
gru_43/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_43/strided_slice_3/stack?
gru_43/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_43/strided_slice_3/stack_1?
gru_43/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_43/strided_slice_3/stack_2?
gru_43/strided_slice_3StridedSlice2gru_43/TensorArrayV2Stack/TensorListStack:tensor:0%gru_43/strided_slice_3/stack:output:0'gru_43/strided_slice_3/stack_1:output:0'gru_43/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru_43/strided_slice_3?
gru_43/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_43/transpose_1/perm?
gru_43/transpose_1	Transpose2gru_43/TensorArrayV2Stack/TensorListStack:tensor:0 gru_43/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
gru_43/transpose_1t
gru_43/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_43/runtime?
dropout_64/IdentityIdentitygru_43/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_64/Identity?
!dense_42/Tensordot/ReadVariableOpReadVariableOp*dense_42_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_42/Tensordot/ReadVariableOp|
dense_42/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_42/Tensordot/axes?
dense_42/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_42/Tensordot/free?
dense_42/Tensordot/ShapeShapedropout_64/Identity:output:0*
T0*
_output_shapes
:2
dense_42/Tensordot/Shape?
 dense_42/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_42/Tensordot/GatherV2/axis?
dense_42/Tensordot/GatherV2GatherV2!dense_42/Tensordot/Shape:output:0 dense_42/Tensordot/free:output:0)dense_42/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_42/Tensordot/GatherV2?
"dense_42/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_42/Tensordot/GatherV2_1/axis?
dense_42/Tensordot/GatherV2_1GatherV2!dense_42/Tensordot/Shape:output:0 dense_42/Tensordot/axes:output:0+dense_42/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_42/Tensordot/GatherV2_1~
dense_42/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_42/Tensordot/Const?
dense_42/Tensordot/ProdProd$dense_42/Tensordot/GatherV2:output:0!dense_42/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_42/Tensordot/Prod?
dense_42/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_42/Tensordot/Const_1?
dense_42/Tensordot/Prod_1Prod&dense_42/Tensordot/GatherV2_1:output:0#dense_42/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_42/Tensordot/Prod_1?
dense_42/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_42/Tensordot/concat/axis?
dense_42/Tensordot/concatConcatV2 dense_42/Tensordot/free:output:0 dense_42/Tensordot/axes:output:0'dense_42/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_42/Tensordot/concat?
dense_42/Tensordot/stackPack dense_42/Tensordot/Prod:output:0"dense_42/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_42/Tensordot/stack?
dense_42/Tensordot/transpose	Transposedropout_64/Identity:output:0"dense_42/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_42/Tensordot/transpose?
dense_42/Tensordot/ReshapeReshape dense_42/Tensordot/transpose:y:0!dense_42/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_42/Tensordot/Reshape?
dense_42/Tensordot/MatMulMatMul#dense_42/Tensordot/Reshape:output:0)dense_42/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_42/Tensordot/MatMul?
dense_42/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_42/Tensordot/Const_2?
 dense_42/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_42/Tensordot/concat_1/axis?
dense_42/Tensordot/concat_1ConcatV2$dense_42/Tensordot/GatherV2:output:0#dense_42/Tensordot/Const_2:output:0)dense_42/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_42/Tensordot/concat_1?
dense_42/TensordotReshape#dense_42/Tensordot/MatMul:product:0$dense_42/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_42/Tensordot?
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_42/BiasAdd/ReadVariableOp?
dense_42/BiasAddBiasAdddense_42/Tensordot:output:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_42/BiasAddx
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_42/Relu?
dropout_65/IdentityIdentitydense_42/Relu:activations:0*
T0*,
_output_shapes
:??????????2
dropout_65/Identity?
!dense_43/Tensordot/ReadVariableOpReadVariableOp*dense_43_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dense_43/Tensordot/ReadVariableOp|
dense_43/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_43/Tensordot/axes?
dense_43/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_43/Tensordot/free?
dense_43/Tensordot/ShapeShapedropout_65/Identity:output:0*
T0*
_output_shapes
:2
dense_43/Tensordot/Shape?
 dense_43/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_43/Tensordot/GatherV2/axis?
dense_43/Tensordot/GatherV2GatherV2!dense_43/Tensordot/Shape:output:0 dense_43/Tensordot/free:output:0)dense_43/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_43/Tensordot/GatherV2?
"dense_43/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_43/Tensordot/GatherV2_1/axis?
dense_43/Tensordot/GatherV2_1GatherV2!dense_43/Tensordot/Shape:output:0 dense_43/Tensordot/axes:output:0+dense_43/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_43/Tensordot/GatherV2_1~
dense_43/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_43/Tensordot/Const?
dense_43/Tensordot/ProdProd$dense_43/Tensordot/GatherV2:output:0!dense_43/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_43/Tensordot/Prod?
dense_43/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_43/Tensordot/Const_1?
dense_43/Tensordot/Prod_1Prod&dense_43/Tensordot/GatherV2_1:output:0#dense_43/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_43/Tensordot/Prod_1?
dense_43/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_43/Tensordot/concat/axis?
dense_43/Tensordot/concatConcatV2 dense_43/Tensordot/free:output:0 dense_43/Tensordot/axes:output:0'dense_43/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_43/Tensordot/concat?
dense_43/Tensordot/stackPack dense_43/Tensordot/Prod:output:0"dense_43/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_43/Tensordot/stack?
dense_43/Tensordot/transpose	Transposedropout_65/Identity:output:0"dense_43/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_43/Tensordot/transpose?
dense_43/Tensordot/ReshapeReshape dense_43/Tensordot/transpose:y:0!dense_43/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_43/Tensordot/Reshape?
dense_43/Tensordot/MatMulMatMul#dense_43/Tensordot/Reshape:output:0)dense_43/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_43/Tensordot/MatMul?
dense_43/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense_43/Tensordot/Const_2?
 dense_43/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_43/Tensordot/concat_1/axis?
dense_43/Tensordot/concat_1ConcatV2$dense_43/Tensordot/GatherV2:output:0#dense_43/Tensordot/Const_2:output:0)dense_43/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_43/Tensordot/concat_1?
dense_43/TensordotReshape#dense_43/Tensordot/MatMul:product:0$dense_43/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2
dense_43/Tensordot?
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_43/BiasAdd/ReadVariableOp?
dense_43/BiasAddBiasAdddense_43/Tensordot:output:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_43/BiasAddx
dense_43/ReluReludense_43/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_43/Relu?
dropout_66/IdentityIdentitydense_43/Relu:activations:0*
T0*,
_output_shapes
:??????????2
dropout_66/Identity?
!dense_44/Tensordot/ReadVariableOpReadVariableOp*dense_44_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_44/Tensordot/ReadVariableOp|
dense_44/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_44/Tensordot/axes?
dense_44/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_44/Tensordot/free?
dense_44/Tensordot/ShapeShapedropout_66/Identity:output:0*
T0*
_output_shapes
:2
dense_44/Tensordot/Shape?
 dense_44/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_44/Tensordot/GatherV2/axis?
dense_44/Tensordot/GatherV2GatherV2!dense_44/Tensordot/Shape:output:0 dense_44/Tensordot/free:output:0)dense_44/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_44/Tensordot/GatherV2?
"dense_44/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_44/Tensordot/GatherV2_1/axis?
dense_44/Tensordot/GatherV2_1GatherV2!dense_44/Tensordot/Shape:output:0 dense_44/Tensordot/axes:output:0+dense_44/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_44/Tensordot/GatherV2_1~
dense_44/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_44/Tensordot/Const?
dense_44/Tensordot/ProdProd$dense_44/Tensordot/GatherV2:output:0!dense_44/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_44/Tensordot/Prod?
dense_44/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_44/Tensordot/Const_1?
dense_44/Tensordot/Prod_1Prod&dense_44/Tensordot/GatherV2_1:output:0#dense_44/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_44/Tensordot/Prod_1?
dense_44/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_44/Tensordot/concat/axis?
dense_44/Tensordot/concatConcatV2 dense_44/Tensordot/free:output:0 dense_44/Tensordot/axes:output:0'dense_44/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_44/Tensordot/concat?
dense_44/Tensordot/stackPack dense_44/Tensordot/Prod:output:0"dense_44/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_44/Tensordot/stack?
dense_44/Tensordot/transpose	Transposedropout_66/Identity:output:0"dense_44/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_44/Tensordot/transpose?
dense_44/Tensordot/ReshapeReshape dense_44/Tensordot/transpose:y:0!dense_44/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_44/Tensordot/Reshape?
dense_44/Tensordot/MatMulMatMul#dense_44/Tensordot/Reshape:output:0)dense_44/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_44/Tensordot/MatMul?
dense_44/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_44/Tensordot/Const_2?
 dense_44/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_44/Tensordot/concat_1/axis?
dense_44/Tensordot/concat_1ConcatV2$dense_44/Tensordot/GatherV2:output:0#dense_44/Tensordot/Const_2:output:0)dense_44/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_44/Tensordot/concat_1?
dense_44/TensordotReshape#dense_44/Tensordot/MatMul:product:0$dense_44/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_44/Tensordot?
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOp?
dense_44/BiasAddBiasAdddense_44/Tensordot:output:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_44/BiasAddx
IdentityIdentitydense_44/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_42/BiasAdd/ReadVariableOp"^dense_42/Tensordot/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp"^dense_43/Tensordot/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp"^dense_44/Tensordot/ReadVariableOp)^gru_42/gru_cell_42/MatMul/ReadVariableOp+^gru_42/gru_cell_42/MatMul_1/ReadVariableOp"^gru_42/gru_cell_42/ReadVariableOp^gru_42/while)^gru_43/gru_cell_43/MatMul/ReadVariableOp+^gru_43/gru_cell_43/MatMul_1/ReadVariableOp"^gru_43/gru_cell_43/ReadVariableOp^gru_43/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2F
!dense_42/Tensordot/ReadVariableOp!dense_42/Tensordot/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2F
!dense_43/Tensordot/ReadVariableOp!dense_43/Tensordot/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2F
!dense_44/Tensordot/ReadVariableOp!dense_44/Tensordot/ReadVariableOp2T
(gru_42/gru_cell_42/MatMul/ReadVariableOp(gru_42/gru_cell_42/MatMul/ReadVariableOp2X
*gru_42/gru_cell_42/MatMul_1/ReadVariableOp*gru_42/gru_cell_42/MatMul_1/ReadVariableOp2F
!gru_42/gru_cell_42/ReadVariableOp!gru_42/gru_cell_42/ReadVariableOp2
gru_42/whilegru_42/while2T
(gru_43/gru_cell_43/MatMul/ReadVariableOp(gru_43/gru_cell_43/MatMul/ReadVariableOp2X
*gru_43/gru_cell_43/MatMul_1/ReadVariableOp*gru_43/gru_cell_43/MatMul_1/ReadVariableOp2F
!gru_43/gru_cell_43/ReadVariableOp!gru_43/gru_cell_43/ReadVariableOp2
gru_43/whilegru_43/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_1576158
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1576158___redundant_placeholder05
1while_while_cond_1576158___redundant_placeholder15
1while_while_cond_1576158___redundant_placeholder25
1while_while_cond_1576158___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
"__inference__wrapped_model_1571916
gru_42_inputK
8sequential_21_gru_42_gru_cell_42_readvariableop_resource:	?R
?sequential_21_gru_42_gru_cell_42_matmul_readvariableop_resource:	?U
Asequential_21_gru_42_gru_cell_42_matmul_1_readvariableop_resource:
??K
8sequential_21_gru_43_gru_cell_43_readvariableop_resource:	?S
?sequential_21_gru_43_gru_cell_43_matmul_readvariableop_resource:
??U
Asequential_21_gru_43_gru_cell_43_matmul_1_readvariableop_resource:
??L
8sequential_21_dense_42_tensordot_readvariableop_resource:
??E
6sequential_21_dense_42_biasadd_readvariableop_resource:	?L
8sequential_21_dense_43_tensordot_readvariableop_resource:
??E
6sequential_21_dense_43_biasadd_readvariableop_resource:	?K
8sequential_21_dense_44_tensordot_readvariableop_resource:	?D
6sequential_21_dense_44_biasadd_readvariableop_resource:
identity??-sequential_21/dense_42/BiasAdd/ReadVariableOp?/sequential_21/dense_42/Tensordot/ReadVariableOp?-sequential_21/dense_43/BiasAdd/ReadVariableOp?/sequential_21/dense_43/Tensordot/ReadVariableOp?-sequential_21/dense_44/BiasAdd/ReadVariableOp?/sequential_21/dense_44/Tensordot/ReadVariableOp?6sequential_21/gru_42/gru_cell_42/MatMul/ReadVariableOp?8sequential_21/gru_42/gru_cell_42/MatMul_1/ReadVariableOp?/sequential_21/gru_42/gru_cell_42/ReadVariableOp?sequential_21/gru_42/while?6sequential_21/gru_43/gru_cell_43/MatMul/ReadVariableOp?8sequential_21/gru_43/gru_cell_43/MatMul_1/ReadVariableOp?/sequential_21/gru_43/gru_cell_43/ReadVariableOp?sequential_21/gru_43/whilet
sequential_21/gru_42/ShapeShapegru_42_input*
T0*
_output_shapes
:2
sequential_21/gru_42/Shape?
(sequential_21/gru_42/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_21/gru_42/strided_slice/stack?
*sequential_21/gru_42/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_21/gru_42/strided_slice/stack_1?
*sequential_21/gru_42/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_21/gru_42/strided_slice/stack_2?
"sequential_21/gru_42/strided_sliceStridedSlice#sequential_21/gru_42/Shape:output:01sequential_21/gru_42/strided_slice/stack:output:03sequential_21/gru_42/strided_slice/stack_1:output:03sequential_21/gru_42/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_21/gru_42/strided_slice?
#sequential_21/gru_42/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_21/gru_42/zeros/packed/1?
!sequential_21/gru_42/zeros/packedPack+sequential_21/gru_42/strided_slice:output:0,sequential_21/gru_42/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_21/gru_42/zeros/packed?
 sequential_21/gru_42/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_21/gru_42/zeros/Const?
sequential_21/gru_42/zerosFill*sequential_21/gru_42/zeros/packed:output:0)sequential_21/gru_42/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_21/gru_42/zeros?
#sequential_21/gru_42/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_21/gru_42/transpose/perm?
sequential_21/gru_42/transpose	Transposegru_42_input,sequential_21/gru_42/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2 
sequential_21/gru_42/transpose?
sequential_21/gru_42/Shape_1Shape"sequential_21/gru_42/transpose:y:0*
T0*
_output_shapes
:2
sequential_21/gru_42/Shape_1?
*sequential_21/gru_42/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_21/gru_42/strided_slice_1/stack?
,sequential_21/gru_42/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_21/gru_42/strided_slice_1/stack_1?
,sequential_21/gru_42/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_21/gru_42/strided_slice_1/stack_2?
$sequential_21/gru_42/strided_slice_1StridedSlice%sequential_21/gru_42/Shape_1:output:03sequential_21/gru_42/strided_slice_1/stack:output:05sequential_21/gru_42/strided_slice_1/stack_1:output:05sequential_21/gru_42/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_21/gru_42/strided_slice_1?
0sequential_21/gru_42/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_21/gru_42/TensorArrayV2/element_shape?
"sequential_21/gru_42/TensorArrayV2TensorListReserve9sequential_21/gru_42/TensorArrayV2/element_shape:output:0-sequential_21/gru_42/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_21/gru_42/TensorArrayV2?
Jsequential_21/gru_42/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jsequential_21/gru_42/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_21/gru_42/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_21/gru_42/transpose:y:0Ssequential_21/gru_42/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_21/gru_42/TensorArrayUnstack/TensorListFromTensor?
*sequential_21/gru_42/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_21/gru_42/strided_slice_2/stack?
,sequential_21/gru_42/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_21/gru_42/strided_slice_2/stack_1?
,sequential_21/gru_42/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_21/gru_42/strided_slice_2/stack_2?
$sequential_21/gru_42/strided_slice_2StridedSlice"sequential_21/gru_42/transpose:y:03sequential_21/gru_42/strided_slice_2/stack:output:05sequential_21/gru_42/strided_slice_2/stack_1:output:05sequential_21/gru_42/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2&
$sequential_21/gru_42/strided_slice_2?
/sequential_21/gru_42/gru_cell_42/ReadVariableOpReadVariableOp8sequential_21_gru_42_gru_cell_42_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_21/gru_42/gru_cell_42/ReadVariableOp?
(sequential_21/gru_42/gru_cell_42/unstackUnpack7sequential_21/gru_42/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2*
(sequential_21/gru_42/gru_cell_42/unstack?
6sequential_21/gru_42/gru_cell_42/MatMul/ReadVariableOpReadVariableOp?sequential_21_gru_42_gru_cell_42_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype028
6sequential_21/gru_42/gru_cell_42/MatMul/ReadVariableOp?
'sequential_21/gru_42/gru_cell_42/MatMulMatMul-sequential_21/gru_42/strided_slice_2:output:0>sequential_21/gru_42/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_21/gru_42/gru_cell_42/MatMul?
(sequential_21/gru_42/gru_cell_42/BiasAddBiasAdd1sequential_21/gru_42/gru_cell_42/MatMul:product:01sequential_21/gru_42/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2*
(sequential_21/gru_42/gru_cell_42/BiasAdd?
0sequential_21/gru_42/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_21/gru_42/gru_cell_42/split/split_dim?
&sequential_21/gru_42/gru_cell_42/splitSplit9sequential_21/gru_42/gru_cell_42/split/split_dim:output:01sequential_21/gru_42/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2(
&sequential_21/gru_42/gru_cell_42/split?
8sequential_21/gru_42/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOpAsequential_21_gru_42_gru_cell_42_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8sequential_21/gru_42/gru_cell_42/MatMul_1/ReadVariableOp?
)sequential_21/gru_42/gru_cell_42/MatMul_1MatMul#sequential_21/gru_42/zeros:output:0@sequential_21/gru_42/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_21/gru_42/gru_cell_42/MatMul_1?
*sequential_21/gru_42/gru_cell_42/BiasAdd_1BiasAdd3sequential_21/gru_42/gru_cell_42/MatMul_1:product:01sequential_21/gru_42/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2,
*sequential_21/gru_42/gru_cell_42/BiasAdd_1?
&sequential_21/gru_42/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2(
&sequential_21/gru_42/gru_cell_42/Const?
2sequential_21/gru_42/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2sequential_21/gru_42/gru_cell_42/split_1/split_dim?
(sequential_21/gru_42/gru_cell_42/split_1SplitV3sequential_21/gru_42/gru_cell_42/BiasAdd_1:output:0/sequential_21/gru_42/gru_cell_42/Const:output:0;sequential_21/gru_42/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2*
(sequential_21/gru_42/gru_cell_42/split_1?
$sequential_21/gru_42/gru_cell_42/addAddV2/sequential_21/gru_42/gru_cell_42/split:output:01sequential_21/gru_42/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2&
$sequential_21/gru_42/gru_cell_42/add?
(sequential_21/gru_42/gru_cell_42/SigmoidSigmoid(sequential_21/gru_42/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_21/gru_42/gru_cell_42/Sigmoid?
&sequential_21/gru_42/gru_cell_42/add_1AddV2/sequential_21/gru_42/gru_cell_42/split:output:11sequential_21/gru_42/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2(
&sequential_21/gru_42/gru_cell_42/add_1?
*sequential_21/gru_42/gru_cell_42/Sigmoid_1Sigmoid*sequential_21/gru_42/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2,
*sequential_21/gru_42/gru_cell_42/Sigmoid_1?
$sequential_21/gru_42/gru_cell_42/mulMul.sequential_21/gru_42/gru_cell_42/Sigmoid_1:y:01sequential_21/gru_42/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2&
$sequential_21/gru_42/gru_cell_42/mul?
&sequential_21/gru_42/gru_cell_42/add_2AddV2/sequential_21/gru_42/gru_cell_42/split:output:2(sequential_21/gru_42/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2(
&sequential_21/gru_42/gru_cell_42/add_2?
%sequential_21/gru_42/gru_cell_42/ReluRelu*sequential_21/gru_42/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2'
%sequential_21/gru_42/gru_cell_42/Relu?
&sequential_21/gru_42/gru_cell_42/mul_1Mul,sequential_21/gru_42/gru_cell_42/Sigmoid:y:0#sequential_21/gru_42/zeros:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_21/gru_42/gru_cell_42/mul_1?
&sequential_21/gru_42/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&sequential_21/gru_42/gru_cell_42/sub/x?
$sequential_21/gru_42/gru_cell_42/subSub/sequential_21/gru_42/gru_cell_42/sub/x:output:0,sequential_21/gru_42/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2&
$sequential_21/gru_42/gru_cell_42/sub?
&sequential_21/gru_42/gru_cell_42/mul_2Mul(sequential_21/gru_42/gru_cell_42/sub:z:03sequential_21/gru_42/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2(
&sequential_21/gru_42/gru_cell_42/mul_2?
&sequential_21/gru_42/gru_cell_42/add_3AddV2*sequential_21/gru_42/gru_cell_42/mul_1:z:0*sequential_21/gru_42/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2(
&sequential_21/gru_42/gru_cell_42/add_3?
2sequential_21/gru_42/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   24
2sequential_21/gru_42/TensorArrayV2_1/element_shape?
$sequential_21/gru_42/TensorArrayV2_1TensorListReserve;sequential_21/gru_42/TensorArrayV2_1/element_shape:output:0-sequential_21/gru_42/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_21/gru_42/TensorArrayV2_1x
sequential_21/gru_42/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_21/gru_42/time?
-sequential_21/gru_42/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_21/gru_42/while/maximum_iterations?
'sequential_21/gru_42/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_21/gru_42/while/loop_counter?
sequential_21/gru_42/whileWhile0sequential_21/gru_42/while/loop_counter:output:06sequential_21/gru_42/while/maximum_iterations:output:0"sequential_21/gru_42/time:output:0-sequential_21/gru_42/TensorArrayV2_1:handle:0#sequential_21/gru_42/zeros:output:0-sequential_21/gru_42/strided_slice_1:output:0Lsequential_21/gru_42/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_21_gru_42_gru_cell_42_readvariableop_resource?sequential_21_gru_42_gru_cell_42_matmul_readvariableop_resourceAsequential_21_gru_42_gru_cell_42_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *3
body+R)
'sequential_21_gru_42_while_body_1571594*3
cond+R)
'sequential_21_gru_42_while_cond_1571593*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
sequential_21/gru_42/while?
Esequential_21/gru_42/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2G
Esequential_21/gru_42/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_21/gru_42/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_21/gru_42/while:output:3Nsequential_21/gru_42/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype029
7sequential_21/gru_42/TensorArrayV2Stack/TensorListStack?
*sequential_21/gru_42/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_21/gru_42/strided_slice_3/stack?
,sequential_21/gru_42/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_21/gru_42/strided_slice_3/stack_1?
,sequential_21/gru_42/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_21/gru_42/strided_slice_3/stack_2?
$sequential_21/gru_42/strided_slice_3StridedSlice@sequential_21/gru_42/TensorArrayV2Stack/TensorListStack:tensor:03sequential_21/gru_42/strided_slice_3/stack:output:05sequential_21/gru_42/strided_slice_3/stack_1:output:05sequential_21/gru_42/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_21/gru_42/strided_slice_3?
%sequential_21/gru_42/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_21/gru_42/transpose_1/perm?
 sequential_21/gru_42/transpose_1	Transpose@sequential_21/gru_42/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_21/gru_42/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_21/gru_42/transpose_1?
sequential_21/gru_42/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_21/gru_42/runtime?
!sequential_21/dropout_63/IdentityIdentity$sequential_21/gru_42/transpose_1:y:0*
T0*,
_output_shapes
:??????????2#
!sequential_21/dropout_63/Identity?
sequential_21/gru_43/ShapeShape*sequential_21/dropout_63/Identity:output:0*
T0*
_output_shapes
:2
sequential_21/gru_43/Shape?
(sequential_21/gru_43/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_21/gru_43/strided_slice/stack?
*sequential_21/gru_43/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_21/gru_43/strided_slice/stack_1?
*sequential_21/gru_43/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_21/gru_43/strided_slice/stack_2?
"sequential_21/gru_43/strided_sliceStridedSlice#sequential_21/gru_43/Shape:output:01sequential_21/gru_43/strided_slice/stack:output:03sequential_21/gru_43/strided_slice/stack_1:output:03sequential_21/gru_43/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_21/gru_43/strided_slice?
#sequential_21/gru_43/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_21/gru_43/zeros/packed/1?
!sequential_21/gru_43/zeros/packedPack+sequential_21/gru_43/strided_slice:output:0,sequential_21/gru_43/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_21/gru_43/zeros/packed?
 sequential_21/gru_43/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_21/gru_43/zeros/Const?
sequential_21/gru_43/zerosFill*sequential_21/gru_43/zeros/packed:output:0)sequential_21/gru_43/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_21/gru_43/zeros?
#sequential_21/gru_43/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_21/gru_43/transpose/perm?
sequential_21/gru_43/transpose	Transpose*sequential_21/dropout_63/Identity:output:0,sequential_21/gru_43/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2 
sequential_21/gru_43/transpose?
sequential_21/gru_43/Shape_1Shape"sequential_21/gru_43/transpose:y:0*
T0*
_output_shapes
:2
sequential_21/gru_43/Shape_1?
*sequential_21/gru_43/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_21/gru_43/strided_slice_1/stack?
,sequential_21/gru_43/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_21/gru_43/strided_slice_1/stack_1?
,sequential_21/gru_43/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_21/gru_43/strided_slice_1/stack_2?
$sequential_21/gru_43/strided_slice_1StridedSlice%sequential_21/gru_43/Shape_1:output:03sequential_21/gru_43/strided_slice_1/stack:output:05sequential_21/gru_43/strided_slice_1/stack_1:output:05sequential_21/gru_43/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_21/gru_43/strided_slice_1?
0sequential_21/gru_43/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_21/gru_43/TensorArrayV2/element_shape?
"sequential_21/gru_43/TensorArrayV2TensorListReserve9sequential_21/gru_43/TensorArrayV2/element_shape:output:0-sequential_21/gru_43/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_21/gru_43/TensorArrayV2?
Jsequential_21/gru_43/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2L
Jsequential_21/gru_43/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_21/gru_43/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_21/gru_43/transpose:y:0Ssequential_21/gru_43/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_21/gru_43/TensorArrayUnstack/TensorListFromTensor?
*sequential_21/gru_43/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_21/gru_43/strided_slice_2/stack?
,sequential_21/gru_43/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_21/gru_43/strided_slice_2/stack_1?
,sequential_21/gru_43/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_21/gru_43/strided_slice_2/stack_2?
$sequential_21/gru_43/strided_slice_2StridedSlice"sequential_21/gru_43/transpose:y:03sequential_21/gru_43/strided_slice_2/stack:output:05sequential_21/gru_43/strided_slice_2/stack_1:output:05sequential_21/gru_43/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_21/gru_43/strided_slice_2?
/sequential_21/gru_43/gru_cell_43/ReadVariableOpReadVariableOp8sequential_21_gru_43_gru_cell_43_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_21/gru_43/gru_cell_43/ReadVariableOp?
(sequential_21/gru_43/gru_cell_43/unstackUnpack7sequential_21/gru_43/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2*
(sequential_21/gru_43/gru_cell_43/unstack?
6sequential_21/gru_43/gru_cell_43/MatMul/ReadVariableOpReadVariableOp?sequential_21_gru_43_gru_cell_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype028
6sequential_21/gru_43/gru_cell_43/MatMul/ReadVariableOp?
'sequential_21/gru_43/gru_cell_43/MatMulMatMul-sequential_21/gru_43/strided_slice_2:output:0>sequential_21/gru_43/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_21/gru_43/gru_cell_43/MatMul?
(sequential_21/gru_43/gru_cell_43/BiasAddBiasAdd1sequential_21/gru_43/gru_cell_43/MatMul:product:01sequential_21/gru_43/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2*
(sequential_21/gru_43/gru_cell_43/BiasAdd?
0sequential_21/gru_43/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_21/gru_43/gru_cell_43/split/split_dim?
&sequential_21/gru_43/gru_cell_43/splitSplit9sequential_21/gru_43/gru_cell_43/split/split_dim:output:01sequential_21/gru_43/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2(
&sequential_21/gru_43/gru_cell_43/split?
8sequential_21/gru_43/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOpAsequential_21_gru_43_gru_cell_43_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8sequential_21/gru_43/gru_cell_43/MatMul_1/ReadVariableOp?
)sequential_21/gru_43/gru_cell_43/MatMul_1MatMul#sequential_21/gru_43/zeros:output:0@sequential_21/gru_43/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_21/gru_43/gru_cell_43/MatMul_1?
*sequential_21/gru_43/gru_cell_43/BiasAdd_1BiasAdd3sequential_21/gru_43/gru_cell_43/MatMul_1:product:01sequential_21/gru_43/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2,
*sequential_21/gru_43/gru_cell_43/BiasAdd_1?
&sequential_21/gru_43/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2(
&sequential_21/gru_43/gru_cell_43/Const?
2sequential_21/gru_43/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2sequential_21/gru_43/gru_cell_43/split_1/split_dim?
(sequential_21/gru_43/gru_cell_43/split_1SplitV3sequential_21/gru_43/gru_cell_43/BiasAdd_1:output:0/sequential_21/gru_43/gru_cell_43/Const:output:0;sequential_21/gru_43/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2*
(sequential_21/gru_43/gru_cell_43/split_1?
$sequential_21/gru_43/gru_cell_43/addAddV2/sequential_21/gru_43/gru_cell_43/split:output:01sequential_21/gru_43/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2&
$sequential_21/gru_43/gru_cell_43/add?
(sequential_21/gru_43/gru_cell_43/SigmoidSigmoid(sequential_21/gru_43/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_21/gru_43/gru_cell_43/Sigmoid?
&sequential_21/gru_43/gru_cell_43/add_1AddV2/sequential_21/gru_43/gru_cell_43/split:output:11sequential_21/gru_43/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2(
&sequential_21/gru_43/gru_cell_43/add_1?
*sequential_21/gru_43/gru_cell_43/Sigmoid_1Sigmoid*sequential_21/gru_43/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2,
*sequential_21/gru_43/gru_cell_43/Sigmoid_1?
$sequential_21/gru_43/gru_cell_43/mulMul.sequential_21/gru_43/gru_cell_43/Sigmoid_1:y:01sequential_21/gru_43/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2&
$sequential_21/gru_43/gru_cell_43/mul?
&sequential_21/gru_43/gru_cell_43/add_2AddV2/sequential_21/gru_43/gru_cell_43/split:output:2(sequential_21/gru_43/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2(
&sequential_21/gru_43/gru_cell_43/add_2?
%sequential_21/gru_43/gru_cell_43/ReluRelu*sequential_21/gru_43/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2'
%sequential_21/gru_43/gru_cell_43/Relu?
&sequential_21/gru_43/gru_cell_43/mul_1Mul,sequential_21/gru_43/gru_cell_43/Sigmoid:y:0#sequential_21/gru_43/zeros:output:0*
T0*(
_output_shapes
:??????????2(
&sequential_21/gru_43/gru_cell_43/mul_1?
&sequential_21/gru_43/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&sequential_21/gru_43/gru_cell_43/sub/x?
$sequential_21/gru_43/gru_cell_43/subSub/sequential_21/gru_43/gru_cell_43/sub/x:output:0,sequential_21/gru_43/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2&
$sequential_21/gru_43/gru_cell_43/sub?
&sequential_21/gru_43/gru_cell_43/mul_2Mul(sequential_21/gru_43/gru_cell_43/sub:z:03sequential_21/gru_43/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2(
&sequential_21/gru_43/gru_cell_43/mul_2?
&sequential_21/gru_43/gru_cell_43/add_3AddV2*sequential_21/gru_43/gru_cell_43/mul_1:z:0*sequential_21/gru_43/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2(
&sequential_21/gru_43/gru_cell_43/add_3?
2sequential_21/gru_43/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   24
2sequential_21/gru_43/TensorArrayV2_1/element_shape?
$sequential_21/gru_43/TensorArrayV2_1TensorListReserve;sequential_21/gru_43/TensorArrayV2_1/element_shape:output:0-sequential_21/gru_43/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_21/gru_43/TensorArrayV2_1x
sequential_21/gru_43/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_21/gru_43/time?
-sequential_21/gru_43/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_21/gru_43/while/maximum_iterations?
'sequential_21/gru_43/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_21/gru_43/while/loop_counter?
sequential_21/gru_43/whileWhile0sequential_21/gru_43/while/loop_counter:output:06sequential_21/gru_43/while/maximum_iterations:output:0"sequential_21/gru_43/time:output:0-sequential_21/gru_43/TensorArrayV2_1:handle:0#sequential_21/gru_43/zeros:output:0-sequential_21/gru_43/strided_slice_1:output:0Lsequential_21/gru_43/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_21_gru_43_gru_cell_43_readvariableop_resource?sequential_21_gru_43_gru_cell_43_matmul_readvariableop_resourceAsequential_21_gru_43_gru_cell_43_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *3
body+R)
'sequential_21_gru_43_while_body_1571744*3
cond+R)
'sequential_21_gru_43_while_cond_1571743*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
sequential_21/gru_43/while?
Esequential_21/gru_43/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2G
Esequential_21/gru_43/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_21/gru_43/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_21/gru_43/while:output:3Nsequential_21/gru_43/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype029
7sequential_21/gru_43/TensorArrayV2Stack/TensorListStack?
*sequential_21/gru_43/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_21/gru_43/strided_slice_3/stack?
,sequential_21/gru_43/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_21/gru_43/strided_slice_3/stack_1?
,sequential_21/gru_43/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_21/gru_43/strided_slice_3/stack_2?
$sequential_21/gru_43/strided_slice_3StridedSlice@sequential_21/gru_43/TensorArrayV2Stack/TensorListStack:tensor:03sequential_21/gru_43/strided_slice_3/stack:output:05sequential_21/gru_43/strided_slice_3/stack_1:output:05sequential_21/gru_43/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_21/gru_43/strided_slice_3?
%sequential_21/gru_43/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_21/gru_43/transpose_1/perm?
 sequential_21/gru_43/transpose_1	Transpose@sequential_21/gru_43/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_21/gru_43/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_21/gru_43/transpose_1?
sequential_21/gru_43/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_21/gru_43/runtime?
!sequential_21/dropout_64/IdentityIdentity$sequential_21/gru_43/transpose_1:y:0*
T0*,
_output_shapes
:??????????2#
!sequential_21/dropout_64/Identity?
/sequential_21/dense_42/Tensordot/ReadVariableOpReadVariableOp8sequential_21_dense_42_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential_21/dense_42/Tensordot/ReadVariableOp?
%sequential_21/dense_42/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_21/dense_42/Tensordot/axes?
%sequential_21/dense_42/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_21/dense_42/Tensordot/free?
&sequential_21/dense_42/Tensordot/ShapeShape*sequential_21/dropout_64/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_21/dense_42/Tensordot/Shape?
.sequential_21/dense_42/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_21/dense_42/Tensordot/GatherV2/axis?
)sequential_21/dense_42/Tensordot/GatherV2GatherV2/sequential_21/dense_42/Tensordot/Shape:output:0.sequential_21/dense_42/Tensordot/free:output:07sequential_21/dense_42/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_21/dense_42/Tensordot/GatherV2?
0sequential_21/dense_42/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_21/dense_42/Tensordot/GatherV2_1/axis?
+sequential_21/dense_42/Tensordot/GatherV2_1GatherV2/sequential_21/dense_42/Tensordot/Shape:output:0.sequential_21/dense_42/Tensordot/axes:output:09sequential_21/dense_42/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_21/dense_42/Tensordot/GatherV2_1?
&sequential_21/dense_42/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_21/dense_42/Tensordot/Const?
%sequential_21/dense_42/Tensordot/ProdProd2sequential_21/dense_42/Tensordot/GatherV2:output:0/sequential_21/dense_42/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_21/dense_42/Tensordot/Prod?
(sequential_21/dense_42/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_21/dense_42/Tensordot/Const_1?
'sequential_21/dense_42/Tensordot/Prod_1Prod4sequential_21/dense_42/Tensordot/GatherV2_1:output:01sequential_21/dense_42/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_21/dense_42/Tensordot/Prod_1?
,sequential_21/dense_42/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_21/dense_42/Tensordot/concat/axis?
'sequential_21/dense_42/Tensordot/concatConcatV2.sequential_21/dense_42/Tensordot/free:output:0.sequential_21/dense_42/Tensordot/axes:output:05sequential_21/dense_42/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_21/dense_42/Tensordot/concat?
&sequential_21/dense_42/Tensordot/stackPack.sequential_21/dense_42/Tensordot/Prod:output:00sequential_21/dense_42/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_21/dense_42/Tensordot/stack?
*sequential_21/dense_42/Tensordot/transpose	Transpose*sequential_21/dropout_64/Identity:output:00sequential_21/dense_42/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2,
*sequential_21/dense_42/Tensordot/transpose?
(sequential_21/dense_42/Tensordot/ReshapeReshape.sequential_21/dense_42/Tensordot/transpose:y:0/sequential_21/dense_42/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_21/dense_42/Tensordot/Reshape?
'sequential_21/dense_42/Tensordot/MatMulMatMul1sequential_21/dense_42/Tensordot/Reshape:output:07sequential_21/dense_42/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_21/dense_42/Tensordot/MatMul?
(sequential_21/dense_42/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2*
(sequential_21/dense_42/Tensordot/Const_2?
.sequential_21/dense_42/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_21/dense_42/Tensordot/concat_1/axis?
)sequential_21/dense_42/Tensordot/concat_1ConcatV22sequential_21/dense_42/Tensordot/GatherV2:output:01sequential_21/dense_42/Tensordot/Const_2:output:07sequential_21/dense_42/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_21/dense_42/Tensordot/concat_1?
 sequential_21/dense_42/TensordotReshape1sequential_21/dense_42/Tensordot/MatMul:product:02sequential_21/dense_42/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_21/dense_42/Tensordot?
-sequential_21/dense_42/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_21/dense_42/BiasAdd/ReadVariableOp?
sequential_21/dense_42/BiasAddBiasAdd)sequential_21/dense_42/Tensordot:output:05sequential_21/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2 
sequential_21/dense_42/BiasAdd?
sequential_21/dense_42/ReluRelu'sequential_21/dense_42/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
sequential_21/dense_42/Relu?
!sequential_21/dropout_65/IdentityIdentity)sequential_21/dense_42/Relu:activations:0*
T0*,
_output_shapes
:??????????2#
!sequential_21/dropout_65/Identity?
/sequential_21/dense_43/Tensordot/ReadVariableOpReadVariableOp8sequential_21_dense_43_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/sequential_21/dense_43/Tensordot/ReadVariableOp?
%sequential_21/dense_43/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_21/dense_43/Tensordot/axes?
%sequential_21/dense_43/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_21/dense_43/Tensordot/free?
&sequential_21/dense_43/Tensordot/ShapeShape*sequential_21/dropout_65/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_21/dense_43/Tensordot/Shape?
.sequential_21/dense_43/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_21/dense_43/Tensordot/GatherV2/axis?
)sequential_21/dense_43/Tensordot/GatherV2GatherV2/sequential_21/dense_43/Tensordot/Shape:output:0.sequential_21/dense_43/Tensordot/free:output:07sequential_21/dense_43/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_21/dense_43/Tensordot/GatherV2?
0sequential_21/dense_43/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_21/dense_43/Tensordot/GatherV2_1/axis?
+sequential_21/dense_43/Tensordot/GatherV2_1GatherV2/sequential_21/dense_43/Tensordot/Shape:output:0.sequential_21/dense_43/Tensordot/axes:output:09sequential_21/dense_43/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_21/dense_43/Tensordot/GatherV2_1?
&sequential_21/dense_43/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_21/dense_43/Tensordot/Const?
%sequential_21/dense_43/Tensordot/ProdProd2sequential_21/dense_43/Tensordot/GatherV2:output:0/sequential_21/dense_43/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_21/dense_43/Tensordot/Prod?
(sequential_21/dense_43/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_21/dense_43/Tensordot/Const_1?
'sequential_21/dense_43/Tensordot/Prod_1Prod4sequential_21/dense_43/Tensordot/GatherV2_1:output:01sequential_21/dense_43/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_21/dense_43/Tensordot/Prod_1?
,sequential_21/dense_43/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_21/dense_43/Tensordot/concat/axis?
'sequential_21/dense_43/Tensordot/concatConcatV2.sequential_21/dense_43/Tensordot/free:output:0.sequential_21/dense_43/Tensordot/axes:output:05sequential_21/dense_43/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_21/dense_43/Tensordot/concat?
&sequential_21/dense_43/Tensordot/stackPack.sequential_21/dense_43/Tensordot/Prod:output:00sequential_21/dense_43/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_21/dense_43/Tensordot/stack?
*sequential_21/dense_43/Tensordot/transpose	Transpose*sequential_21/dropout_65/Identity:output:00sequential_21/dense_43/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2,
*sequential_21/dense_43/Tensordot/transpose?
(sequential_21/dense_43/Tensordot/ReshapeReshape.sequential_21/dense_43/Tensordot/transpose:y:0/sequential_21/dense_43/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_21/dense_43/Tensordot/Reshape?
'sequential_21/dense_43/Tensordot/MatMulMatMul1sequential_21/dense_43/Tensordot/Reshape:output:07sequential_21/dense_43/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_21/dense_43/Tensordot/MatMul?
(sequential_21/dense_43/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2*
(sequential_21/dense_43/Tensordot/Const_2?
.sequential_21/dense_43/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_21/dense_43/Tensordot/concat_1/axis?
)sequential_21/dense_43/Tensordot/concat_1ConcatV22sequential_21/dense_43/Tensordot/GatherV2:output:01sequential_21/dense_43/Tensordot/Const_2:output:07sequential_21/dense_43/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_21/dense_43/Tensordot/concat_1?
 sequential_21/dense_43/TensordotReshape1sequential_21/dense_43/Tensordot/MatMul:product:02sequential_21/dense_43/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_21/dense_43/Tensordot?
-sequential_21/dense_43/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_21/dense_43/BiasAdd/ReadVariableOp?
sequential_21/dense_43/BiasAddBiasAdd)sequential_21/dense_43/Tensordot:output:05sequential_21/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2 
sequential_21/dense_43/BiasAdd?
sequential_21/dense_43/ReluRelu'sequential_21/dense_43/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
sequential_21/dense_43/Relu?
!sequential_21/dropout_66/IdentityIdentity)sequential_21/dense_43/Relu:activations:0*
T0*,
_output_shapes
:??????????2#
!sequential_21/dropout_66/Identity?
/sequential_21/dense_44/Tensordot/ReadVariableOpReadVariableOp8sequential_21_dense_44_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_21/dense_44/Tensordot/ReadVariableOp?
%sequential_21/dense_44/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_21/dense_44/Tensordot/axes?
%sequential_21/dense_44/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_21/dense_44/Tensordot/free?
&sequential_21/dense_44/Tensordot/ShapeShape*sequential_21/dropout_66/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_21/dense_44/Tensordot/Shape?
.sequential_21/dense_44/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_21/dense_44/Tensordot/GatherV2/axis?
)sequential_21/dense_44/Tensordot/GatherV2GatherV2/sequential_21/dense_44/Tensordot/Shape:output:0.sequential_21/dense_44/Tensordot/free:output:07sequential_21/dense_44/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_21/dense_44/Tensordot/GatherV2?
0sequential_21/dense_44/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_21/dense_44/Tensordot/GatherV2_1/axis?
+sequential_21/dense_44/Tensordot/GatherV2_1GatherV2/sequential_21/dense_44/Tensordot/Shape:output:0.sequential_21/dense_44/Tensordot/axes:output:09sequential_21/dense_44/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_21/dense_44/Tensordot/GatherV2_1?
&sequential_21/dense_44/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_21/dense_44/Tensordot/Const?
%sequential_21/dense_44/Tensordot/ProdProd2sequential_21/dense_44/Tensordot/GatherV2:output:0/sequential_21/dense_44/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_21/dense_44/Tensordot/Prod?
(sequential_21/dense_44/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_21/dense_44/Tensordot/Const_1?
'sequential_21/dense_44/Tensordot/Prod_1Prod4sequential_21/dense_44/Tensordot/GatherV2_1:output:01sequential_21/dense_44/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_21/dense_44/Tensordot/Prod_1?
,sequential_21/dense_44/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_21/dense_44/Tensordot/concat/axis?
'sequential_21/dense_44/Tensordot/concatConcatV2.sequential_21/dense_44/Tensordot/free:output:0.sequential_21/dense_44/Tensordot/axes:output:05sequential_21/dense_44/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_21/dense_44/Tensordot/concat?
&sequential_21/dense_44/Tensordot/stackPack.sequential_21/dense_44/Tensordot/Prod:output:00sequential_21/dense_44/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_21/dense_44/Tensordot/stack?
*sequential_21/dense_44/Tensordot/transpose	Transpose*sequential_21/dropout_66/Identity:output:00sequential_21/dense_44/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2,
*sequential_21/dense_44/Tensordot/transpose?
(sequential_21/dense_44/Tensordot/ReshapeReshape.sequential_21/dense_44/Tensordot/transpose:y:0/sequential_21/dense_44/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_21/dense_44/Tensordot/Reshape?
'sequential_21/dense_44/Tensordot/MatMulMatMul1sequential_21/dense_44/Tensordot/Reshape:output:07sequential_21/dense_44/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'sequential_21/dense_44/Tensordot/MatMul?
(sequential_21/dense_44/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_21/dense_44/Tensordot/Const_2?
.sequential_21/dense_44/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_21/dense_44/Tensordot/concat_1/axis?
)sequential_21/dense_44/Tensordot/concat_1ConcatV22sequential_21/dense_44/Tensordot/GatherV2:output:01sequential_21/dense_44/Tensordot/Const_2:output:07sequential_21/dense_44/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_21/dense_44/Tensordot/concat_1?
 sequential_21/dense_44/TensordotReshape1sequential_21/dense_44/Tensordot/MatMul:product:02sequential_21/dense_44/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2"
 sequential_21/dense_44/Tensordot?
-sequential_21/dense_44/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_21/dense_44/BiasAdd/ReadVariableOp?
sequential_21/dense_44/BiasAddBiasAdd)sequential_21/dense_44/Tensordot:output:05sequential_21/dense_44/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 
sequential_21/dense_44/BiasAdd?
IdentityIdentity'sequential_21/dense_44/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp.^sequential_21/dense_42/BiasAdd/ReadVariableOp0^sequential_21/dense_42/Tensordot/ReadVariableOp.^sequential_21/dense_43/BiasAdd/ReadVariableOp0^sequential_21/dense_43/Tensordot/ReadVariableOp.^sequential_21/dense_44/BiasAdd/ReadVariableOp0^sequential_21/dense_44/Tensordot/ReadVariableOp7^sequential_21/gru_42/gru_cell_42/MatMul/ReadVariableOp9^sequential_21/gru_42/gru_cell_42/MatMul_1/ReadVariableOp0^sequential_21/gru_42/gru_cell_42/ReadVariableOp^sequential_21/gru_42/while7^sequential_21/gru_43/gru_cell_43/MatMul/ReadVariableOp9^sequential_21/gru_43/gru_cell_43/MatMul_1/ReadVariableOp0^sequential_21/gru_43/gru_cell_43/ReadVariableOp^sequential_21/gru_43/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2^
-sequential_21/dense_42/BiasAdd/ReadVariableOp-sequential_21/dense_42/BiasAdd/ReadVariableOp2b
/sequential_21/dense_42/Tensordot/ReadVariableOp/sequential_21/dense_42/Tensordot/ReadVariableOp2^
-sequential_21/dense_43/BiasAdd/ReadVariableOp-sequential_21/dense_43/BiasAdd/ReadVariableOp2b
/sequential_21/dense_43/Tensordot/ReadVariableOp/sequential_21/dense_43/Tensordot/ReadVariableOp2^
-sequential_21/dense_44/BiasAdd/ReadVariableOp-sequential_21/dense_44/BiasAdd/ReadVariableOp2b
/sequential_21/dense_44/Tensordot/ReadVariableOp/sequential_21/dense_44/Tensordot/ReadVariableOp2p
6sequential_21/gru_42/gru_cell_42/MatMul/ReadVariableOp6sequential_21/gru_42/gru_cell_42/MatMul/ReadVariableOp2t
8sequential_21/gru_42/gru_cell_42/MatMul_1/ReadVariableOp8sequential_21/gru_42/gru_cell_42/MatMul_1/ReadVariableOp2b
/sequential_21/gru_42/gru_cell_42/ReadVariableOp/sequential_21/gru_42/gru_cell_42/ReadVariableOp28
sequential_21/gru_42/whilesequential_21/gru_42/while2p
6sequential_21/gru_43/gru_cell_43/MatMul/ReadVariableOp6sequential_21/gru_43/gru_cell_43/MatMul/ReadVariableOp2t
8sequential_21/gru_43/gru_cell_43/MatMul_1/ReadVariableOp8sequential_21/gru_43/gru_cell_43/MatMul_1/ReadVariableOp2b
/sequential_21/gru_43/gru_cell_43/ReadVariableOp/sequential_21/gru_43/gru_cell_43/ReadVariableOp28
sequential_21/gru_43/whilesequential_21/gru_43/while:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_42_input
?*
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1574174
gru_42_input!
gru_42_1574140:	?!
gru_42_1574142:	?"
gru_42_1574144:
??!
gru_43_1574148:	?"
gru_43_1574150:
??"
gru_43_1574152:
??$
dense_42_1574156:
??
dense_42_1574158:	?$
dense_43_1574162:
??
dense_43_1574164:	?#
dense_44_1574168:	?
dense_44_1574170:
identity?? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall?gru_42/StatefulPartitionedCall?gru_43/StatefulPartitionedCall?
gru_42/StatefulPartitionedCallStatefulPartitionedCallgru_42_inputgru_42_1574140gru_42_1574142gru_42_1574144*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_42_layer_call_and_return_conditional_losses_15732072 
gru_42/StatefulPartitionedCall?
dropout_63/PartitionedCallPartitionedCall'gru_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_63_layer_call_and_return_conditional_losses_15732202
dropout_63/PartitionedCall?
gru_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_63/PartitionedCall:output:0gru_43_1574148gru_43_1574150gru_43_1574152*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_43_layer_call_and_return_conditional_losses_15733742 
gru_43/StatefulPartitionedCall?
dropout_64/PartitionedCallPartitionedCall'gru_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_64_layer_call_and_return_conditional_losses_15733872
dropout_64/PartitionedCall?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall#dropout_64/PartitionedCall:output:0dense_42_1574156dense_42_1574158*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_15734202"
 dense_42/StatefulPartitionedCall?
dropout_65/PartitionedCallPartitionedCall)dense_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_65_layer_call_and_return_conditional_losses_15734312
dropout_65/PartitionedCall?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_65/PartitionedCall:output:0dense_43_1574162dense_43_1574164*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_15734642"
 dense_43/StatefulPartitionedCall?
dropout_66/PartitionedCallPartitionedCall)dense_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_66_layer_call_and_return_conditional_losses_15734752
dropout_66/PartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall#dropout_66/PartitionedCall:output:0dense_44_1574168dense_44_1574170*
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
E__inference_dense_44_layer_call_and_return_conditional_losses_15735072"
 dense_44/StatefulPartitionedCall?
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall^gru_42/StatefulPartitionedCall^gru_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2@
gru_42/StatefulPartitionedCallgru_42/StatefulPartitionedCall2@
gru_43/StatefulPartitionedCallgru_43/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_42_input
?
?
while_cond_1575322
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1575322___redundant_placeholder05
1while_while_cond_1575322___redundant_placeholder15
1while_while_cond_1575322___redundant_placeholder25
1while_while_cond_1575322___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?
e
G__inference_dropout_63_layer_call_and_return_conditional_losses_1575767

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_21_layer_call_fn_1574137
gru_42_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_42_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
J__inference_sequential_21_layer_call_and_return_conditional_losses_15740812
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
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_42_input
?
?
(__inference_gru_43_layer_call_fn_1576434

inputs
unknown:	?
	unknown_0:
??
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_43_layer_call_and_return_conditional_losses_15733742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?X
?
C__inference_gru_42_layer_call_and_return_conditional_losses_1575565

inputs6
#gru_cell_42_readvariableop_resource:	?=
*gru_cell_42_matmul_readvariableop_resource:	?@
,gru_cell_42_matmul_1_readvariableop_resource:
??
identity??!gru_cell_42/MatMul/ReadVariableOp?#gru_cell_42/MatMul_1/ReadVariableOp?gru_cell_42/ReadVariableOp?whileD
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
B :?2
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
:??????????2
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
:?????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_42/ReadVariableOpReadVariableOp#gru_cell_42_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_42/ReadVariableOp?
gru_cell_42/unstackUnpack"gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_42/unstack?
!gru_cell_42/MatMul/ReadVariableOpReadVariableOp*gru_cell_42_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_42/MatMul/ReadVariableOp?
gru_cell_42/MatMulMatMulstrided_slice_2:output:0)gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul?
gru_cell_42/BiasAddBiasAddgru_cell_42/MatMul:product:0gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd?
gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split/split_dim?
gru_cell_42/splitSplit$gru_cell_42/split/split_dim:output:0gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split?
#gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_42_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_42/MatMul_1/ReadVariableOp?
gru_cell_42/MatMul_1MatMulzeros:output:0+gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul_1?
gru_cell_42/BiasAdd_1BiasAddgru_cell_42/MatMul_1:product:0gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd_1{
gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_42/Const?
gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split_1/split_dim?
gru_cell_42/split_1SplitVgru_cell_42/BiasAdd_1:output:0gru_cell_42/Const:output:0&gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split_1?
gru_cell_42/addAddV2gru_cell_42/split:output:0gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add}
gru_cell_42/SigmoidSigmoidgru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid?
gru_cell_42/add_1AddV2gru_cell_42/split:output:1gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_1?
gru_cell_42/Sigmoid_1Sigmoidgru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid_1?
gru_cell_42/mulMulgru_cell_42/Sigmoid_1:y:0gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul?
gru_cell_42/add_2AddV2gru_cell_42/split:output:2gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_2v
gru_cell_42/ReluRelugru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Relu?
gru_cell_42/mul_1Mulgru_cell_42/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_1k
gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_42/sub/x?
gru_cell_42/subSubgru_cell_42/sub/x:output:0gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/sub?
gru_cell_42/mul_2Mulgru_cell_42/sub:z:0gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_2?
gru_cell_42/add_3AddV2gru_cell_42/mul_1:z:0gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_42_readvariableop_resource*gru_cell_42_matmul_readvariableop_resource,gru_cell_42_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1575476*
condR
while_cond_1575475*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
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
:??????????2
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
:??????????2

Identity?
NoOpNoOp"^gru_cell_42/MatMul/ReadVariableOp$^gru_cell_42/MatMul_1/ReadVariableOp^gru_cell_42/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_42/MatMul/ReadVariableOp!gru_cell_42/MatMul/ReadVariableOp2J
#gru_cell_42/MatMul_1/ReadVariableOp#gru_cell_42/MatMul_1/ReadVariableOp28
gru_cell_42/ReadVariableOpgru_cell_42/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_66_layer_call_and_return_conditional_losses_1573571

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_63_layer_call_and_return_conditional_losses_1575779

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_1572564
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1572564___redundant_placeholder05
1while_while_cond_1572564___redundant_placeholder15
1while_while_cond_1572564___redundant_placeholder25
1while_while_cond_1572564___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?E
?
while_body_1573915
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_42_readvariableop_resource_0:	?E
2while_gru_cell_42_matmul_readvariableop_resource_0:	?H
4while_gru_cell_42_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_42_readvariableop_resource:	?C
0while_gru_cell_42_matmul_readvariableop_resource:	?F
2while_gru_cell_42_matmul_1_readvariableop_resource:
????'while/gru_cell_42/MatMul/ReadVariableOp?)while/gru_cell_42/MatMul_1/ReadVariableOp? while/gru_cell_42/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_42/ReadVariableOpReadVariableOp+while_gru_cell_42_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_42/ReadVariableOp?
while/gru_cell_42/unstackUnpack(while/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_42/unstack?
'while/gru_cell_42/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_42/MatMul/ReadVariableOp?
while/gru_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul?
while/gru_cell_42/BiasAddBiasAdd"while/gru_cell_42/MatMul:product:0"while/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd?
!while/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_42/split/split_dim?
while/gru_cell_42/splitSplit*while/gru_cell_42/split/split_dim:output:0"while/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split?
)while/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_42_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_42/MatMul_1/ReadVariableOp?
while/gru_cell_42/MatMul_1MatMulwhile_placeholder_21while/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul_1?
while/gru_cell_42/BiasAdd_1BiasAdd$while/gru_cell_42/MatMul_1:product:0"while/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd_1?
while/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_42/Const?
#while/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_42/split_1/split_dim?
while/gru_cell_42/split_1SplitV$while/gru_cell_42/BiasAdd_1:output:0 while/gru_cell_42/Const:output:0,while/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split_1?
while/gru_cell_42/addAddV2 while/gru_cell_42/split:output:0"while/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add?
while/gru_cell_42/SigmoidSigmoidwhile/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid?
while/gru_cell_42/add_1AddV2 while/gru_cell_42/split:output:1"while/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_1?
while/gru_cell_42/Sigmoid_1Sigmoidwhile/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid_1?
while/gru_cell_42/mulMulwhile/gru_cell_42/Sigmoid_1:y:0"while/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul?
while/gru_cell_42/add_2AddV2 while/gru_cell_42/split:output:2while/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_2?
while/gru_cell_42/ReluReluwhile/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Relu?
while/gru_cell_42/mul_1Mulwhile/gru_cell_42/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_1w
while/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_42/sub/x?
while/gru_cell_42/subSub while/gru_cell_42/sub/x:output:0while/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/sub?
while/gru_cell_42/mul_2Mulwhile/gru_cell_42/sub:z:0$while/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_2?
while/gru_cell_42/add_3AddV2while/gru_cell_42/mul_1:z:0while/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_42/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_42/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_42/MatMul/ReadVariableOp*^while/gru_cell_42/MatMul_1/ReadVariableOp!^while/gru_cell_42/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_42_matmul_1_readvariableop_resource4while_gru_cell_42_matmul_1_readvariableop_resource_0"f
0while_gru_cell_42_matmul_readvariableop_resource2while_gru_cell_42_matmul_readvariableop_resource_0"X
)while_gru_cell_42_readvariableop_resource+while_gru_cell_42_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_42/MatMul/ReadVariableOp'while/gru_cell_42/MatMul/ReadVariableOp2V
)while/gru_cell_42/MatMul_1/ReadVariableOp)while/gru_cell_42/MatMul_1/ReadVariableOp2D
 while/gru_cell_42/ReadVariableOp while/gru_cell_42/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?E
?
while_body_1573285
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_43_readvariableop_resource_0:	?F
2while_gru_cell_43_matmul_readvariableop_resource_0:
??H
4while_gru_cell_43_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_43_readvariableop_resource:	?D
0while_gru_cell_43_matmul_readvariableop_resource:
??F
2while_gru_cell_43_matmul_1_readvariableop_resource:
????'while/gru_cell_43/MatMul/ReadVariableOp?)while/gru_cell_43/MatMul_1/ReadVariableOp? while/gru_cell_43/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_43/ReadVariableOpReadVariableOp+while_gru_cell_43_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_43/ReadVariableOp?
while/gru_cell_43/unstackUnpack(while/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_43/unstack?
'while/gru_cell_43/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_43_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_43/MatMul/ReadVariableOp?
while/gru_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul?
while/gru_cell_43/BiasAddBiasAdd"while/gru_cell_43/MatMul:product:0"while/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd?
!while/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_43/split/split_dim?
while/gru_cell_43/splitSplit*while/gru_cell_43/split/split_dim:output:0"while/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split?
)while/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_43_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_43/MatMul_1/ReadVariableOp?
while/gru_cell_43/MatMul_1MatMulwhile_placeholder_21while/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul_1?
while/gru_cell_43/BiasAdd_1BiasAdd$while/gru_cell_43/MatMul_1:product:0"while/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd_1?
while/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_43/Const?
#while/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_43/split_1/split_dim?
while/gru_cell_43/split_1SplitV$while/gru_cell_43/BiasAdd_1:output:0 while/gru_cell_43/Const:output:0,while/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split_1?
while/gru_cell_43/addAddV2 while/gru_cell_43/split:output:0"while/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add?
while/gru_cell_43/SigmoidSigmoidwhile/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid?
while/gru_cell_43/add_1AddV2 while/gru_cell_43/split:output:1"while/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_1?
while/gru_cell_43/Sigmoid_1Sigmoidwhile/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid_1?
while/gru_cell_43/mulMulwhile/gru_cell_43/Sigmoid_1:y:0"while/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul?
while/gru_cell_43/add_2AddV2 while/gru_cell_43/split:output:2while/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_2?
while/gru_cell_43/ReluReluwhile/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Relu?
while/gru_cell_43/mul_1Mulwhile/gru_cell_43/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_1w
while/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_43/sub/x?
while/gru_cell_43/subSub while/gru_cell_43/sub/x:output:0while/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/sub?
while/gru_cell_43/mul_2Mulwhile/gru_cell_43/sub:z:0$while/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_2?
while/gru_cell_43/add_3AddV2while/gru_cell_43/mul_1:z:0while/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_43/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_43/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_43/MatMul/ReadVariableOp*^while/gru_cell_43/MatMul_1/ReadVariableOp!^while/gru_cell_43/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_43_matmul_1_readvariableop_resource4while_gru_cell_43_matmul_1_readvariableop_resource_0"f
0while_gru_cell_43_matmul_readvariableop_resource2while_gru_cell_43_matmul_readvariableop_resource_0"X
)while_gru_cell_43_readvariableop_resource+while_gru_cell_43_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_43/MatMul/ReadVariableOp'while/gru_cell_43/MatMul/ReadVariableOp2V
)while/gru_cell_43/MatMul_1/ReadVariableOp)while/gru_cell_43/MatMul_1/ReadVariableOp2D
 while/gru_cell_43/ReadVariableOp while/gru_cell_43/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?
f
G__inference_dropout_66_layer_call_and_return_conditional_losses_1576596

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_1575475
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1575475___redundant_placeholder05
1while_while_cond_1575475___redundant_placeholder15
1while_while_cond_1575475___redundant_placeholder25
1while_while_cond_1575475___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?
f
G__inference_dropout_64_layer_call_and_return_conditional_losses_1576462

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
E__inference_dense_44_layer_call_and_return_conditional_losses_1576636

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
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
:??????????2
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_44_layer_call_fn_1576645

inputs
unknown:	?
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
E__inference_dense_44_layer_call_and_return_conditional_losses_15735072
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_64_layer_call_and_return_conditional_losses_1573387

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_65_layer_call_and_return_conditional_losses_1576529

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?Y
?
C__inference_gru_43_layer_call_and_return_conditional_losses_1576095
inputs_06
#gru_cell_43_readvariableop_resource:	?>
*gru_cell_43_matmul_readvariableop_resource:
??@
,gru_cell_43_matmul_1_readvariableop_resource:
??
identity??!gru_cell_43/MatMul/ReadVariableOp?#gru_cell_43/MatMul_1/ReadVariableOp?gru_cell_43/ReadVariableOp?whileF
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
B :?2
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
:??????????2
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
!:???????????????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_43/ReadVariableOpReadVariableOp#gru_cell_43_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_43/ReadVariableOp?
gru_cell_43/unstackUnpack"gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_43/unstack?
!gru_cell_43/MatMul/ReadVariableOpReadVariableOp*gru_cell_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_43/MatMul/ReadVariableOp?
gru_cell_43/MatMulMatMulstrided_slice_2:output:0)gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul?
gru_cell_43/BiasAddBiasAddgru_cell_43/MatMul:product:0gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd?
gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split/split_dim?
gru_cell_43/splitSplit$gru_cell_43/split/split_dim:output:0gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split?
#gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_43_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_43/MatMul_1/ReadVariableOp?
gru_cell_43/MatMul_1MatMulzeros:output:0+gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul_1?
gru_cell_43/BiasAdd_1BiasAddgru_cell_43/MatMul_1:product:0gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd_1{
gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_43/Const?
gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split_1/split_dim?
gru_cell_43/split_1SplitVgru_cell_43/BiasAdd_1:output:0gru_cell_43/Const:output:0&gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split_1?
gru_cell_43/addAddV2gru_cell_43/split:output:0gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add}
gru_cell_43/SigmoidSigmoidgru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid?
gru_cell_43/add_1AddV2gru_cell_43/split:output:1gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_1?
gru_cell_43/Sigmoid_1Sigmoidgru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid_1?
gru_cell_43/mulMulgru_cell_43/Sigmoid_1:y:0gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul?
gru_cell_43/add_2AddV2gru_cell_43/split:output:2gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_2v
gru_cell_43/ReluRelugru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Relu?
gru_cell_43/mul_1Mulgru_cell_43/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_1k
gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_43/sub/x?
gru_cell_43/subSubgru_cell_43/sub/x:output:0gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/sub?
gru_cell_43/mul_2Mulgru_cell_43/sub:z:0gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_2?
gru_cell_43/add_3AddV2gru_cell_43/mul_1:z:0gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_43_readvariableop_resource*gru_cell_43_matmul_readvariableop_resource,gru_cell_43_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1576006*
condR
while_cond_1576005*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
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
:??????????*
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
!:???????????????????2
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
!:???????????????????2

Identity?
NoOpNoOp"^gru_cell_43/MatMul/ReadVariableOp$^gru_cell_43/MatMul_1/ReadVariableOp^gru_cell_43/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2F
!gru_cell_43/MatMul/ReadVariableOp!gru_cell_43/MatMul/ReadVariableOp2J
#gru_cell_43/MatMul_1/ReadVariableOp#gru_cell_43/MatMul_1/ReadVariableOp28
gru_cell_43/ReadVariableOpgru_cell_43/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_1573716
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1573716___redundant_placeholder05
1while_while_cond_1573716___redundant_placeholder15
1while_while_cond_1573716___redundant_placeholder25
1while_while_cond_1573716___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?E
?
while_body_1575323
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_42_readvariableop_resource_0:	?E
2while_gru_cell_42_matmul_readvariableop_resource_0:	?H
4while_gru_cell_42_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_42_readvariableop_resource:	?C
0while_gru_cell_42_matmul_readvariableop_resource:	?F
2while_gru_cell_42_matmul_1_readvariableop_resource:
????'while/gru_cell_42/MatMul/ReadVariableOp?)while/gru_cell_42/MatMul_1/ReadVariableOp? while/gru_cell_42/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_42/ReadVariableOpReadVariableOp+while_gru_cell_42_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_42/ReadVariableOp?
while/gru_cell_42/unstackUnpack(while/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_42/unstack?
'while/gru_cell_42/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_42/MatMul/ReadVariableOp?
while/gru_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul?
while/gru_cell_42/BiasAddBiasAdd"while/gru_cell_42/MatMul:product:0"while/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd?
!while/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_42/split/split_dim?
while/gru_cell_42/splitSplit*while/gru_cell_42/split/split_dim:output:0"while/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split?
)while/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_42_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_42/MatMul_1/ReadVariableOp?
while/gru_cell_42/MatMul_1MatMulwhile_placeholder_21while/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul_1?
while/gru_cell_42/BiasAdd_1BiasAdd$while/gru_cell_42/MatMul_1:product:0"while/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd_1?
while/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_42/Const?
#while/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_42/split_1/split_dim?
while/gru_cell_42/split_1SplitV$while/gru_cell_42/BiasAdd_1:output:0 while/gru_cell_42/Const:output:0,while/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split_1?
while/gru_cell_42/addAddV2 while/gru_cell_42/split:output:0"while/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add?
while/gru_cell_42/SigmoidSigmoidwhile/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid?
while/gru_cell_42/add_1AddV2 while/gru_cell_42/split:output:1"while/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_1?
while/gru_cell_42/Sigmoid_1Sigmoidwhile/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid_1?
while/gru_cell_42/mulMulwhile/gru_cell_42/Sigmoid_1:y:0"while/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul?
while/gru_cell_42/add_2AddV2 while/gru_cell_42/split:output:2while/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_2?
while/gru_cell_42/ReluReluwhile/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Relu?
while/gru_cell_42/mul_1Mulwhile/gru_cell_42/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_1w
while/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_42/sub/x?
while/gru_cell_42/subSub while/gru_cell_42/sub/x:output:0while/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/sub?
while/gru_cell_42/mul_2Mulwhile/gru_cell_42/sub:z:0$while/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_2?
while/gru_cell_42/add_3AddV2while/gru_cell_42/mul_1:z:0while/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_42/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_42/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_42/MatMul/ReadVariableOp*^while/gru_cell_42/MatMul_1/ReadVariableOp!^while/gru_cell_42/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_42_matmul_1_readvariableop_resource4while_gru_cell_42_matmul_1_readvariableop_resource_0"f
0while_gru_cell_42_matmul_readvariableop_resource2while_gru_cell_42_matmul_readvariableop_resource_0"X
)while_gru_cell_42_readvariableop_resource+while_gru_cell_42_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_42/MatMul/ReadVariableOp'while/gru_cell_42/MatMul/ReadVariableOp2V
)while/gru_cell_42/MatMul_1/ReadVariableOp)while/gru_cell_42/MatMul_1/ReadVariableOp2D
 while/gru_cell_42/ReadVariableOp while/gru_cell_42/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?X
?
C__inference_gru_43_layer_call_and_return_conditional_losses_1576248

inputs6
#gru_cell_43_readvariableop_resource:	?>
*gru_cell_43_matmul_readvariableop_resource:
??@
,gru_cell_43_matmul_1_readvariableop_resource:
??
identity??!gru_cell_43/MatMul/ReadVariableOp?#gru_cell_43/MatMul_1/ReadVariableOp?gru_cell_43/ReadVariableOp?whileD
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
B :?2
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
:??????????2
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
:??????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_43/ReadVariableOpReadVariableOp#gru_cell_43_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_43/ReadVariableOp?
gru_cell_43/unstackUnpack"gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_43/unstack?
!gru_cell_43/MatMul/ReadVariableOpReadVariableOp*gru_cell_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_43/MatMul/ReadVariableOp?
gru_cell_43/MatMulMatMulstrided_slice_2:output:0)gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul?
gru_cell_43/BiasAddBiasAddgru_cell_43/MatMul:product:0gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd?
gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split/split_dim?
gru_cell_43/splitSplit$gru_cell_43/split/split_dim:output:0gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split?
#gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_43_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_43/MatMul_1/ReadVariableOp?
gru_cell_43/MatMul_1MatMulzeros:output:0+gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul_1?
gru_cell_43/BiasAdd_1BiasAddgru_cell_43/MatMul_1:product:0gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd_1{
gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_43/Const?
gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split_1/split_dim?
gru_cell_43/split_1SplitVgru_cell_43/BiasAdd_1:output:0gru_cell_43/Const:output:0&gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split_1?
gru_cell_43/addAddV2gru_cell_43/split:output:0gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add}
gru_cell_43/SigmoidSigmoidgru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid?
gru_cell_43/add_1AddV2gru_cell_43/split:output:1gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_1?
gru_cell_43/Sigmoid_1Sigmoidgru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid_1?
gru_cell_43/mulMulgru_cell_43/Sigmoid_1:y:0gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul?
gru_cell_43/add_2AddV2gru_cell_43/split:output:2gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_2v
gru_cell_43/ReluRelugru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Relu?
gru_cell_43/mul_1Mulgru_cell_43/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_1k
gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_43/sub/x?
gru_cell_43/subSubgru_cell_43/sub/x:output:0gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/sub?
gru_cell_43/mul_2Mulgru_cell_43/sub:z:0gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_2?
gru_cell_43/add_3AddV2gru_cell_43/mul_1:z:0gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_43_readvariableop_resource*gru_cell_43_matmul_readvariableop_resource,gru_cell_43_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1576159*
condR
while_cond_1576158*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
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
:??????????2
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
:??????????2

Identity?
NoOpNoOp"^gru_cell_43/MatMul/ReadVariableOp$^gru_cell_43/MatMul_1/ReadVariableOp^gru_cell_43/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_43/MatMul/ReadVariableOp!gru_cell_43/MatMul/ReadVariableOp2J
#gru_cell_43/MatMul_1/ReadVariableOp#gru_cell_43/MatMul_1/ReadVariableOp28
gru_cell_43/ReadVariableOpgru_cell_43/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?f
?
'sequential_21_gru_43_while_body_1571744F
Bsequential_21_gru_43_while_sequential_21_gru_43_while_loop_counterL
Hsequential_21_gru_43_while_sequential_21_gru_43_while_maximum_iterations*
&sequential_21_gru_43_while_placeholder,
(sequential_21_gru_43_while_placeholder_1,
(sequential_21_gru_43_while_placeholder_2E
Asequential_21_gru_43_while_sequential_21_gru_43_strided_slice_1_0?
}sequential_21_gru_43_while_tensorarrayv2read_tensorlistgetitem_sequential_21_gru_43_tensorarrayunstack_tensorlistfromtensor_0S
@sequential_21_gru_43_while_gru_cell_43_readvariableop_resource_0:	?[
Gsequential_21_gru_43_while_gru_cell_43_matmul_readvariableop_resource_0:
??]
Isequential_21_gru_43_while_gru_cell_43_matmul_1_readvariableop_resource_0:
??'
#sequential_21_gru_43_while_identity)
%sequential_21_gru_43_while_identity_1)
%sequential_21_gru_43_while_identity_2)
%sequential_21_gru_43_while_identity_3)
%sequential_21_gru_43_while_identity_4C
?sequential_21_gru_43_while_sequential_21_gru_43_strided_slice_1
{sequential_21_gru_43_while_tensorarrayv2read_tensorlistgetitem_sequential_21_gru_43_tensorarrayunstack_tensorlistfromtensorQ
>sequential_21_gru_43_while_gru_cell_43_readvariableop_resource:	?Y
Esequential_21_gru_43_while_gru_cell_43_matmul_readvariableop_resource:
??[
Gsequential_21_gru_43_while_gru_cell_43_matmul_1_readvariableop_resource:
????<sequential_21/gru_43/while/gru_cell_43/MatMul/ReadVariableOp?>sequential_21/gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp?5sequential_21/gru_43/while/gru_cell_43/ReadVariableOp?
Lsequential_21/gru_43/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2N
Lsequential_21/gru_43/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_21/gru_43/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_21_gru_43_while_tensorarrayv2read_tensorlistgetitem_sequential_21_gru_43_tensorarrayunstack_tensorlistfromtensor_0&sequential_21_gru_43_while_placeholderUsequential_21/gru_43/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02@
>sequential_21/gru_43/while/TensorArrayV2Read/TensorListGetItem?
5sequential_21/gru_43/while/gru_cell_43/ReadVariableOpReadVariableOp@sequential_21_gru_43_while_gru_cell_43_readvariableop_resource_0*
_output_shapes
:	?*
dtype027
5sequential_21/gru_43/while/gru_cell_43/ReadVariableOp?
.sequential_21/gru_43/while/gru_cell_43/unstackUnpack=sequential_21/gru_43/while/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num20
.sequential_21/gru_43/while/gru_cell_43/unstack?
<sequential_21/gru_43/while/gru_cell_43/MatMul/ReadVariableOpReadVariableOpGsequential_21_gru_43_while_gru_cell_43_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02>
<sequential_21/gru_43/while/gru_cell_43/MatMul/ReadVariableOp?
-sequential_21/gru_43/while/gru_cell_43/MatMulMatMulEsequential_21/gru_43/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential_21/gru_43/while/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_21/gru_43/while/gru_cell_43/MatMul?
.sequential_21/gru_43/while/gru_cell_43/BiasAddBiasAdd7sequential_21/gru_43/while/gru_cell_43/MatMul:product:07sequential_21/gru_43/while/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????20
.sequential_21/gru_43/while/gru_cell_43/BiasAdd?
6sequential_21/gru_43/while/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_21/gru_43/while/gru_cell_43/split/split_dim?
,sequential_21/gru_43/while/gru_cell_43/splitSplit?sequential_21/gru_43/while/gru_cell_43/split/split_dim:output:07sequential_21/gru_43/while/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2.
,sequential_21/gru_43/while/gru_cell_43/split?
>sequential_21/gru_43/while/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOpIsequential_21_gru_43_while_gru_cell_43_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02@
>sequential_21/gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp?
/sequential_21/gru_43/while/gru_cell_43/MatMul_1MatMul(sequential_21_gru_43_while_placeholder_2Fsequential_21/gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_21/gru_43/while/gru_cell_43/MatMul_1?
0sequential_21/gru_43/while/gru_cell_43/BiasAdd_1BiasAdd9sequential_21/gru_43/while/gru_cell_43/MatMul_1:product:07sequential_21/gru_43/while/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????22
0sequential_21/gru_43/while/gru_cell_43/BiasAdd_1?
,sequential_21/gru_43/while/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2.
,sequential_21/gru_43/while/gru_cell_43/Const?
8sequential_21/gru_43/while/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8sequential_21/gru_43/while/gru_cell_43/split_1/split_dim?
.sequential_21/gru_43/while/gru_cell_43/split_1SplitV9sequential_21/gru_43/while/gru_cell_43/BiasAdd_1:output:05sequential_21/gru_43/while/gru_cell_43/Const:output:0Asequential_21/gru_43/while/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split20
.sequential_21/gru_43/while/gru_cell_43/split_1?
*sequential_21/gru_43/while/gru_cell_43/addAddV25sequential_21/gru_43/while/gru_cell_43/split:output:07sequential_21/gru_43/while/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2,
*sequential_21/gru_43/while/gru_cell_43/add?
.sequential_21/gru_43/while/gru_cell_43/SigmoidSigmoid.sequential_21/gru_43/while/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????20
.sequential_21/gru_43/while/gru_cell_43/Sigmoid?
,sequential_21/gru_43/while/gru_cell_43/add_1AddV25sequential_21/gru_43/while/gru_cell_43/split:output:17sequential_21/gru_43/while/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2.
,sequential_21/gru_43/while/gru_cell_43/add_1?
0sequential_21/gru_43/while/gru_cell_43/Sigmoid_1Sigmoid0sequential_21/gru_43/while/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????22
0sequential_21/gru_43/while/gru_cell_43/Sigmoid_1?
*sequential_21/gru_43/while/gru_cell_43/mulMul4sequential_21/gru_43/while/gru_cell_43/Sigmoid_1:y:07sequential_21/gru_43/while/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2,
*sequential_21/gru_43/while/gru_cell_43/mul?
,sequential_21/gru_43/while/gru_cell_43/add_2AddV25sequential_21/gru_43/while/gru_cell_43/split:output:2.sequential_21/gru_43/while/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2.
,sequential_21/gru_43/while/gru_cell_43/add_2?
+sequential_21/gru_43/while/gru_cell_43/ReluRelu0sequential_21/gru_43/while/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2-
+sequential_21/gru_43/while/gru_cell_43/Relu?
,sequential_21/gru_43/while/gru_cell_43/mul_1Mul2sequential_21/gru_43/while/gru_cell_43/Sigmoid:y:0(sequential_21_gru_43_while_placeholder_2*
T0*(
_output_shapes
:??????????2.
,sequential_21/gru_43/while/gru_cell_43/mul_1?
,sequential_21/gru_43/while/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,sequential_21/gru_43/while/gru_cell_43/sub/x?
*sequential_21/gru_43/while/gru_cell_43/subSub5sequential_21/gru_43/while/gru_cell_43/sub/x:output:02sequential_21/gru_43/while/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2,
*sequential_21/gru_43/while/gru_cell_43/sub?
,sequential_21/gru_43/while/gru_cell_43/mul_2Mul.sequential_21/gru_43/while/gru_cell_43/sub:z:09sequential_21/gru_43/while/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,sequential_21/gru_43/while/gru_cell_43/mul_2?
,sequential_21/gru_43/while/gru_cell_43/add_3AddV20sequential_21/gru_43/while/gru_cell_43/mul_1:z:00sequential_21/gru_43/while/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2.
,sequential_21/gru_43/while/gru_cell_43/add_3?
?sequential_21/gru_43/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_21_gru_43_while_placeholder_1&sequential_21_gru_43_while_placeholder0sequential_21/gru_43/while/gru_cell_43/add_3:z:0*
_output_shapes
: *
element_dtype02A
?sequential_21/gru_43/while/TensorArrayV2Write/TensorListSetItem?
 sequential_21/gru_43/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_21/gru_43/while/add/y?
sequential_21/gru_43/while/addAddV2&sequential_21_gru_43_while_placeholder)sequential_21/gru_43/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_21/gru_43/while/add?
"sequential_21/gru_43/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_21/gru_43/while/add_1/y?
 sequential_21/gru_43/while/add_1AddV2Bsequential_21_gru_43_while_sequential_21_gru_43_while_loop_counter+sequential_21/gru_43/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_21/gru_43/while/add_1?
#sequential_21/gru_43/while/IdentityIdentity$sequential_21/gru_43/while/add_1:z:0 ^sequential_21/gru_43/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_21/gru_43/while/Identity?
%sequential_21/gru_43/while/Identity_1IdentityHsequential_21_gru_43_while_sequential_21_gru_43_while_maximum_iterations ^sequential_21/gru_43/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_21/gru_43/while/Identity_1?
%sequential_21/gru_43/while/Identity_2Identity"sequential_21/gru_43/while/add:z:0 ^sequential_21/gru_43/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_21/gru_43/while/Identity_2?
%sequential_21/gru_43/while/Identity_3IdentityOsequential_21/gru_43/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_21/gru_43/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_21/gru_43/while/Identity_3?
%sequential_21/gru_43/while/Identity_4Identity0sequential_21/gru_43/while/gru_cell_43/add_3:z:0 ^sequential_21/gru_43/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_21/gru_43/while/Identity_4?
sequential_21/gru_43/while/NoOpNoOp=^sequential_21/gru_43/while/gru_cell_43/MatMul/ReadVariableOp?^sequential_21/gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp6^sequential_21/gru_43/while/gru_cell_43/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_21/gru_43/while/NoOp"?
Gsequential_21_gru_43_while_gru_cell_43_matmul_1_readvariableop_resourceIsequential_21_gru_43_while_gru_cell_43_matmul_1_readvariableop_resource_0"?
Esequential_21_gru_43_while_gru_cell_43_matmul_readvariableop_resourceGsequential_21_gru_43_while_gru_cell_43_matmul_readvariableop_resource_0"?
>sequential_21_gru_43_while_gru_cell_43_readvariableop_resource@sequential_21_gru_43_while_gru_cell_43_readvariableop_resource_0"S
#sequential_21_gru_43_while_identity,sequential_21/gru_43/while/Identity:output:0"W
%sequential_21_gru_43_while_identity_1.sequential_21/gru_43/while/Identity_1:output:0"W
%sequential_21_gru_43_while_identity_2.sequential_21/gru_43/while/Identity_2:output:0"W
%sequential_21_gru_43_while_identity_3.sequential_21/gru_43/while/Identity_3:output:0"W
%sequential_21_gru_43_while_identity_4.sequential_21/gru_43/while/Identity_4:output:0"?
?sequential_21_gru_43_while_sequential_21_gru_43_strided_slice_1Asequential_21_gru_43_while_sequential_21_gru_43_strided_slice_1_0"?
{sequential_21_gru_43_while_tensorarrayv2read_tensorlistgetitem_sequential_21_gru_43_tensorarrayunstack_tensorlistfromtensor}sequential_21_gru_43_while_tensorarrayv2read_tensorlistgetitem_sequential_21_gru_43_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2|
<sequential_21/gru_43/while/gru_cell_43/MatMul/ReadVariableOp<sequential_21/gru_43/while/gru_cell_43/MatMul/ReadVariableOp2?
>sequential_21/gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp>sequential_21/gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp2n
5sequential_21/gru_43/while/gru_cell_43/ReadVariableOp5sequential_21/gru_43/while/gru_cell_43/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?
H
,__inference_dropout_64_layer_call_fn_1576467

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_64_layer_call_and_return_conditional_losses_15733872
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_1572552

inputs

states*
readvariableop_resource:	?2
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
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
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
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
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
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
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

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
.:??????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
e
,__inference_dropout_65_layer_call_fn_1576539

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_65_layer_call_and_return_conditional_losses_15736042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?X
?
C__inference_gru_43_layer_call_and_return_conditional_losses_1573374

inputs6
#gru_cell_43_readvariableop_resource:	?>
*gru_cell_43_matmul_readvariableop_resource:
??@
,gru_cell_43_matmul_1_readvariableop_resource:
??
identity??!gru_cell_43/MatMul/ReadVariableOp?#gru_cell_43/MatMul_1/ReadVariableOp?gru_cell_43/ReadVariableOp?whileD
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
B :?2
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
:??????????2
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
:??????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_43/ReadVariableOpReadVariableOp#gru_cell_43_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_43/ReadVariableOp?
gru_cell_43/unstackUnpack"gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_43/unstack?
!gru_cell_43/MatMul/ReadVariableOpReadVariableOp*gru_cell_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_43/MatMul/ReadVariableOp?
gru_cell_43/MatMulMatMulstrided_slice_2:output:0)gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul?
gru_cell_43/BiasAddBiasAddgru_cell_43/MatMul:product:0gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd?
gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split/split_dim?
gru_cell_43/splitSplit$gru_cell_43/split/split_dim:output:0gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split?
#gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_43_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_43/MatMul_1/ReadVariableOp?
gru_cell_43/MatMul_1MatMulzeros:output:0+gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul_1?
gru_cell_43/BiasAdd_1BiasAddgru_cell_43/MatMul_1:product:0gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd_1{
gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_43/Const?
gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split_1/split_dim?
gru_cell_43/split_1SplitVgru_cell_43/BiasAdd_1:output:0gru_cell_43/Const:output:0&gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split_1?
gru_cell_43/addAddV2gru_cell_43/split:output:0gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add}
gru_cell_43/SigmoidSigmoidgru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid?
gru_cell_43/add_1AddV2gru_cell_43/split:output:1gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_1?
gru_cell_43/Sigmoid_1Sigmoidgru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid_1?
gru_cell_43/mulMulgru_cell_43/Sigmoid_1:y:0gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul?
gru_cell_43/add_2AddV2gru_cell_43/split:output:2gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_2v
gru_cell_43/ReluRelugru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Relu?
gru_cell_43/mul_1Mulgru_cell_43/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_1k
gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_43/sub/x?
gru_cell_43/subSubgru_cell_43/sub/x:output:0gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/sub?
gru_cell_43/mul_2Mulgru_cell_43/sub:z:0gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_2?
gru_cell_43/add_3AddV2gru_cell_43/mul_1:z:0gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_43_readvariableop_resource*gru_cell_43_matmul_readvariableop_resource,gru_cell_43_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1573285*
condR
while_cond_1573284*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
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
:??????????2
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
:??????????2

Identity?
NoOpNoOp"^gru_cell_43/MatMul/ReadVariableOp$^gru_cell_43/MatMul_1/ReadVariableOp^gru_cell_43/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_43/MatMul/ReadVariableOp!gru_cell_43/MatMul/ReadVariableOp2J
#gru_cell_43/MatMul_1/ReadVariableOp#gru_cell_43/MatMul_1/ReadVariableOp28
gru_cell_43/ReadVariableOpgru_cell_43/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'sequential_21_gru_42_while_cond_1571593F
Bsequential_21_gru_42_while_sequential_21_gru_42_while_loop_counterL
Hsequential_21_gru_42_while_sequential_21_gru_42_while_maximum_iterations*
&sequential_21_gru_42_while_placeholder,
(sequential_21_gru_42_while_placeholder_1,
(sequential_21_gru_42_while_placeholder_2H
Dsequential_21_gru_42_while_less_sequential_21_gru_42_strided_slice_1_
[sequential_21_gru_42_while_sequential_21_gru_42_while_cond_1571593___redundant_placeholder0_
[sequential_21_gru_42_while_sequential_21_gru_42_while_cond_1571593___redundant_placeholder1_
[sequential_21_gru_42_while_sequential_21_gru_42_while_cond_1571593___redundant_placeholder2_
[sequential_21_gru_42_while_sequential_21_gru_42_while_cond_1571593___redundant_placeholder3'
#sequential_21_gru_42_while_identity
?
sequential_21/gru_42/while/LessLess&sequential_21_gru_42_while_placeholderDsequential_21_gru_42_while_less_sequential_21_gru_42_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_21/gru_42/while/Less?
#sequential_21/gru_42/while/IdentityIdentity#sequential_21/gru_42/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_21/gru_42/while/Identity"S
#sequential_21_gru_42_while_identity,sequential_21/gru_42/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?"
?
while_body_1571999
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_42_1572021_0:	?.
while_gru_cell_42_1572023_0:	?/
while_gru_cell_42_1572025_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_42_1572021:	?,
while_gru_cell_42_1572023:	?-
while_gru_cell_42_1572025:
????)while/gru_cell_42/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_42/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_42_1572021_0while_gru_cell_42_1572023_0while_gru_cell_42_1572025_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_15719862+
)while/gru_cell_42/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_42/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_42/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_42/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"8
while_gru_cell_42_1572021while_gru_cell_42_1572021_0"8
while_gru_cell_42_1572023while_gru_cell_42_1572023_0"8
while_gru_cell_42_1572025while_gru_cell_42_1572025_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2V
)while/gru_cell_42/StatefulPartitionedCall)while/gru_cell_42/StatefulPartitionedCall: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_dense_43_layer_call_fn_1576579

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_15734642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
E__inference_dense_42_layer_call_and_return_conditional_losses_1573420

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
G__inference_dropout_65_layer_call_and_return_conditional_losses_1573604

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
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
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
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_gru_42_layer_call_fn_1575762

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_42_layer_call_and_return_conditional_losses_15740042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
while_body_1573717
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_43_readvariableop_resource_0:	?F
2while_gru_cell_43_matmul_readvariableop_resource_0:
??H
4while_gru_cell_43_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_43_readvariableop_resource:	?D
0while_gru_cell_43_matmul_readvariableop_resource:
??F
2while_gru_cell_43_matmul_1_readvariableop_resource:
????'while/gru_cell_43/MatMul/ReadVariableOp?)while/gru_cell_43/MatMul_1/ReadVariableOp? while/gru_cell_43/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_43/ReadVariableOpReadVariableOp+while_gru_cell_43_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_43/ReadVariableOp?
while/gru_cell_43/unstackUnpack(while/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_43/unstack?
'while/gru_cell_43/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_43_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_43/MatMul/ReadVariableOp?
while/gru_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul?
while/gru_cell_43/BiasAddBiasAdd"while/gru_cell_43/MatMul:product:0"while/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd?
!while/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_43/split/split_dim?
while/gru_cell_43/splitSplit*while/gru_cell_43/split/split_dim:output:0"while/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split?
)while/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_43_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_43/MatMul_1/ReadVariableOp?
while/gru_cell_43/MatMul_1MatMulwhile_placeholder_21while/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul_1?
while/gru_cell_43/BiasAdd_1BiasAdd$while/gru_cell_43/MatMul_1:product:0"while/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd_1?
while/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_43/Const?
#while/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_43/split_1/split_dim?
while/gru_cell_43/split_1SplitV$while/gru_cell_43/BiasAdd_1:output:0 while/gru_cell_43/Const:output:0,while/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split_1?
while/gru_cell_43/addAddV2 while/gru_cell_43/split:output:0"while/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add?
while/gru_cell_43/SigmoidSigmoidwhile/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid?
while/gru_cell_43/add_1AddV2 while/gru_cell_43/split:output:1"while/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_1?
while/gru_cell_43/Sigmoid_1Sigmoidwhile/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid_1?
while/gru_cell_43/mulMulwhile/gru_cell_43/Sigmoid_1:y:0"while/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul?
while/gru_cell_43/add_2AddV2 while/gru_cell_43/split:output:2while/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_2?
while/gru_cell_43/ReluReluwhile/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Relu?
while/gru_cell_43/mul_1Mulwhile/gru_cell_43/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_1w
while/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_43/sub/x?
while/gru_cell_43/subSub while/gru_cell_43/sub/x:output:0while/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/sub?
while/gru_cell_43/mul_2Mulwhile/gru_cell_43/sub:z:0$while/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_2?
while/gru_cell_43/add_3AddV2while/gru_cell_43/mul_1:z:0while/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_43/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_43/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_43/MatMul/ReadVariableOp*^while/gru_cell_43/MatMul_1/ReadVariableOp!^while/gru_cell_43/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_43_matmul_1_readvariableop_resource4while_gru_cell_43_matmul_1_readvariableop_resource_0"f
0while_gru_cell_43_matmul_readvariableop_resource2while_gru_cell_43_matmul_readvariableop_resource_0"X
)while_gru_cell_43_readvariableop_resource+while_gru_cell_43_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_43/MatMul/ReadVariableOp'while/gru_cell_43/MatMul/ReadVariableOp2V
)while/gru_cell_43/MatMul_1/ReadVariableOp)while/gru_cell_43/MatMul_1/ReadVariableOp2D
 while/gru_cell_43/ReadVariableOp while/gru_cell_43/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?"
?
while_body_1572192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_42_1572214_0:	?.
while_gru_cell_42_1572216_0:	?/
while_gru_cell_42_1572218_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_42_1572214:	?,
while_gru_cell_42_1572216:	?-
while_gru_cell_42_1572218:
????)while/gru_cell_42/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_42/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_42_1572214_0while_gru_cell_42_1572216_0while_gru_cell_42_1572218_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_15721292+
)while/gru_cell_42/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_42/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_42/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_42/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"8
while_gru_cell_42_1572214while_gru_cell_42_1572214_0"8
while_gru_cell_42_1572216while_gru_cell_42_1572216_0"8
while_gru_cell_42_1572218while_gru_cell_42_1572218_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2V
)while/gru_cell_42/StatefulPartitionedCall)while/gru_cell_42/StatefulPartitionedCall: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?P
?	
gru_43_while_body_1574855*
&gru_43_while_gru_43_while_loop_counter0
,gru_43_while_gru_43_while_maximum_iterations
gru_43_while_placeholder
gru_43_while_placeholder_1
gru_43_while_placeholder_2)
%gru_43_while_gru_43_strided_slice_1_0e
agru_43_while_tensorarrayv2read_tensorlistgetitem_gru_43_tensorarrayunstack_tensorlistfromtensor_0E
2gru_43_while_gru_cell_43_readvariableop_resource_0:	?M
9gru_43_while_gru_cell_43_matmul_readvariableop_resource_0:
??O
;gru_43_while_gru_cell_43_matmul_1_readvariableop_resource_0:
??
gru_43_while_identity
gru_43_while_identity_1
gru_43_while_identity_2
gru_43_while_identity_3
gru_43_while_identity_4'
#gru_43_while_gru_43_strided_slice_1c
_gru_43_while_tensorarrayv2read_tensorlistgetitem_gru_43_tensorarrayunstack_tensorlistfromtensorC
0gru_43_while_gru_cell_43_readvariableop_resource:	?K
7gru_43_while_gru_cell_43_matmul_readvariableop_resource:
??M
9gru_43_while_gru_cell_43_matmul_1_readvariableop_resource:
????.gru_43/while/gru_cell_43/MatMul/ReadVariableOp?0gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp?'gru_43/while/gru_cell_43/ReadVariableOp?
>gru_43/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2@
>gru_43/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_43/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_43_while_tensorarrayv2read_tensorlistgetitem_gru_43_tensorarrayunstack_tensorlistfromtensor_0gru_43_while_placeholderGgru_43/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype022
0gru_43/while/TensorArrayV2Read/TensorListGetItem?
'gru_43/while/gru_cell_43/ReadVariableOpReadVariableOp2gru_43_while_gru_cell_43_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_43/while/gru_cell_43/ReadVariableOp?
 gru_43/while/gru_cell_43/unstackUnpack/gru_43/while/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_43/while/gru_cell_43/unstack?
.gru_43/while/gru_cell_43/MatMul/ReadVariableOpReadVariableOp9gru_43_while_gru_cell_43_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype020
.gru_43/while/gru_cell_43/MatMul/ReadVariableOp?
gru_43/while/gru_cell_43/MatMulMatMul7gru_43/while/TensorArrayV2Read/TensorListGetItem:item:06gru_43/while/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_43/while/gru_cell_43/MatMul?
 gru_43/while/gru_cell_43/BiasAddBiasAdd)gru_43/while/gru_cell_43/MatMul:product:0)gru_43/while/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_43/while/gru_cell_43/BiasAdd?
(gru_43/while/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_43/while/gru_cell_43/split/split_dim?
gru_43/while/gru_cell_43/splitSplit1gru_43/while/gru_cell_43/split/split_dim:output:0)gru_43/while/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_43/while/gru_cell_43/split?
0gru_43/while/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp;gru_43_while_gru_cell_43_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp?
!gru_43/while/gru_cell_43/MatMul_1MatMulgru_43_while_placeholder_28gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_43/while/gru_cell_43/MatMul_1?
"gru_43/while/gru_cell_43/BiasAdd_1BiasAdd+gru_43/while/gru_cell_43/MatMul_1:product:0)gru_43/while/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_43/while/gru_cell_43/BiasAdd_1?
gru_43/while/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2 
gru_43/while/gru_cell_43/Const?
*gru_43/while/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_43/while/gru_cell_43/split_1/split_dim?
 gru_43/while/gru_cell_43/split_1SplitV+gru_43/while/gru_cell_43/BiasAdd_1:output:0'gru_43/while/gru_cell_43/Const:output:03gru_43/while/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_43/while/gru_cell_43/split_1?
gru_43/while/gru_cell_43/addAddV2'gru_43/while/gru_cell_43/split:output:0)gru_43/while/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_43/while/gru_cell_43/add?
 gru_43/while/gru_cell_43/SigmoidSigmoid gru_43/while/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_43/while/gru_cell_43/Sigmoid?
gru_43/while/gru_cell_43/add_1AddV2'gru_43/while/gru_cell_43/split:output:1)gru_43/while/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_43/while/gru_cell_43/add_1?
"gru_43/while/gru_cell_43/Sigmoid_1Sigmoid"gru_43/while/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_43/while/gru_cell_43/Sigmoid_1?
gru_43/while/gru_cell_43/mulMul&gru_43/while/gru_cell_43/Sigmoid_1:y:0)gru_43/while/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_43/while/gru_cell_43/mul?
gru_43/while/gru_cell_43/add_2AddV2'gru_43/while/gru_cell_43/split:output:2 gru_43/while/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_43/while/gru_cell_43/add_2?
gru_43/while/gru_cell_43/ReluRelu"gru_43/while/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_43/while/gru_cell_43/Relu?
gru_43/while/gru_cell_43/mul_1Mul$gru_43/while/gru_cell_43/Sigmoid:y:0gru_43_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_43/while/gru_cell_43/mul_1?
gru_43/while/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_43/while/gru_cell_43/sub/x?
gru_43/while/gru_cell_43/subSub'gru_43/while/gru_cell_43/sub/x:output:0$gru_43/while/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_43/while/gru_cell_43/sub?
gru_43/while/gru_cell_43/mul_2Mul gru_43/while/gru_cell_43/sub:z:0+gru_43/while/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_43/while/gru_cell_43/mul_2?
gru_43/while/gru_cell_43/add_3AddV2"gru_43/while/gru_cell_43/mul_1:z:0"gru_43/while/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_43/while/gru_cell_43/add_3?
1gru_43/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_43_while_placeholder_1gru_43_while_placeholder"gru_43/while/gru_cell_43/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_43/while/TensorArrayV2Write/TensorListSetItemj
gru_43/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_43/while/add/y?
gru_43/while/addAddV2gru_43_while_placeholdergru_43/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_43/while/addn
gru_43/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_43/while/add_1/y?
gru_43/while/add_1AddV2&gru_43_while_gru_43_while_loop_countergru_43/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_43/while/add_1?
gru_43/while/IdentityIdentitygru_43/while/add_1:z:0^gru_43/while/NoOp*
T0*
_output_shapes
: 2
gru_43/while/Identity?
gru_43/while/Identity_1Identity,gru_43_while_gru_43_while_maximum_iterations^gru_43/while/NoOp*
T0*
_output_shapes
: 2
gru_43/while/Identity_1?
gru_43/while/Identity_2Identitygru_43/while/add:z:0^gru_43/while/NoOp*
T0*
_output_shapes
: 2
gru_43/while/Identity_2?
gru_43/while/Identity_3IdentityAgru_43/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_43/while/NoOp*
T0*
_output_shapes
: 2
gru_43/while/Identity_3?
gru_43/while/Identity_4Identity"gru_43/while/gru_cell_43/add_3:z:0^gru_43/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_43/while/Identity_4?
gru_43/while/NoOpNoOp/^gru_43/while/gru_cell_43/MatMul/ReadVariableOp1^gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp(^gru_43/while/gru_cell_43/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_43/while/NoOp"L
#gru_43_while_gru_43_strided_slice_1%gru_43_while_gru_43_strided_slice_1_0"x
9gru_43_while_gru_cell_43_matmul_1_readvariableop_resource;gru_43_while_gru_cell_43_matmul_1_readvariableop_resource_0"t
7gru_43_while_gru_cell_43_matmul_readvariableop_resource9gru_43_while_gru_cell_43_matmul_readvariableop_resource_0"f
0gru_43_while_gru_cell_43_readvariableop_resource2gru_43_while_gru_cell_43_readvariableop_resource_0"7
gru_43_while_identitygru_43/while/Identity:output:0";
gru_43_while_identity_1 gru_43/while/Identity_1:output:0";
gru_43_while_identity_2 gru_43/while/Identity_2:output:0";
gru_43_while_identity_3 gru_43/while/Identity_3:output:0";
gru_43_while_identity_4 gru_43/while/Identity_4:output:0"?
_gru_43_while_tensorarrayv2read_tensorlistgetitem_gru_43_tensorarrayunstack_tensorlistfromtensoragru_43_while_tensorarrayv2read_tensorlistgetitem_gru_43_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_43/while/gru_cell_43/MatMul/ReadVariableOp.gru_43/while/gru_cell_43/MatMul/ReadVariableOp2d
0gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp0gru_43/while/gru_cell_43/MatMul_1/ReadVariableOp2R
'gru_43/while/gru_cell_43/ReadVariableOp'gru_43/while/gru_cell_43/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?P
?	
gru_42_while_body_1574698*
&gru_42_while_gru_42_while_loop_counter0
,gru_42_while_gru_42_while_maximum_iterations
gru_42_while_placeholder
gru_42_while_placeholder_1
gru_42_while_placeholder_2)
%gru_42_while_gru_42_strided_slice_1_0e
agru_42_while_tensorarrayv2read_tensorlistgetitem_gru_42_tensorarrayunstack_tensorlistfromtensor_0E
2gru_42_while_gru_cell_42_readvariableop_resource_0:	?L
9gru_42_while_gru_cell_42_matmul_readvariableop_resource_0:	?O
;gru_42_while_gru_cell_42_matmul_1_readvariableop_resource_0:
??
gru_42_while_identity
gru_42_while_identity_1
gru_42_while_identity_2
gru_42_while_identity_3
gru_42_while_identity_4'
#gru_42_while_gru_42_strided_slice_1c
_gru_42_while_tensorarrayv2read_tensorlistgetitem_gru_42_tensorarrayunstack_tensorlistfromtensorC
0gru_42_while_gru_cell_42_readvariableop_resource:	?J
7gru_42_while_gru_cell_42_matmul_readvariableop_resource:	?M
9gru_42_while_gru_cell_42_matmul_1_readvariableop_resource:
????.gru_42/while/gru_cell_42/MatMul/ReadVariableOp?0gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp?'gru_42/while/gru_cell_42/ReadVariableOp?
>gru_42/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_42/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_42/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_42_while_tensorarrayv2read_tensorlistgetitem_gru_42_tensorarrayunstack_tensorlistfromtensor_0gru_42_while_placeholderGgru_42/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_42/while/TensorArrayV2Read/TensorListGetItem?
'gru_42/while/gru_cell_42/ReadVariableOpReadVariableOp2gru_42_while_gru_cell_42_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_42/while/gru_cell_42/ReadVariableOp?
 gru_42/while/gru_cell_42/unstackUnpack/gru_42/while/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_42/while/gru_cell_42/unstack?
.gru_42/while/gru_cell_42/MatMul/ReadVariableOpReadVariableOp9gru_42_while_gru_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_42/while/gru_cell_42/MatMul/ReadVariableOp?
gru_42/while/gru_cell_42/MatMulMatMul7gru_42/while/TensorArrayV2Read/TensorListGetItem:item:06gru_42/while/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_42/while/gru_cell_42/MatMul?
 gru_42/while/gru_cell_42/BiasAddBiasAdd)gru_42/while/gru_cell_42/MatMul:product:0)gru_42/while/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_42/while/gru_cell_42/BiasAdd?
(gru_42/while/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_42/while/gru_cell_42/split/split_dim?
gru_42/while/gru_cell_42/splitSplit1gru_42/while/gru_cell_42/split/split_dim:output:0)gru_42/while/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2 
gru_42/while/gru_cell_42/split?
0gru_42/while/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp;gru_42_while_gru_cell_42_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp?
!gru_42/while/gru_cell_42/MatMul_1MatMulgru_42_while_placeholder_28gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_42/while/gru_cell_42/MatMul_1?
"gru_42/while/gru_cell_42/BiasAdd_1BiasAdd+gru_42/while/gru_cell_42/MatMul_1:product:0)gru_42/while/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_42/while/gru_cell_42/BiasAdd_1?
gru_42/while/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2 
gru_42/while/gru_cell_42/Const?
*gru_42/while/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_42/while/gru_cell_42/split_1/split_dim?
 gru_42/while/gru_cell_42/split_1SplitV+gru_42/while/gru_cell_42/BiasAdd_1:output:0'gru_42/while/gru_cell_42/Const:output:03gru_42/while/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2"
 gru_42/while/gru_cell_42/split_1?
gru_42/while/gru_cell_42/addAddV2'gru_42/while/gru_cell_42/split:output:0)gru_42/while/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_42/while/gru_cell_42/add?
 gru_42/while/gru_cell_42/SigmoidSigmoid gru_42/while/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2"
 gru_42/while/gru_cell_42/Sigmoid?
gru_42/while/gru_cell_42/add_1AddV2'gru_42/while/gru_cell_42/split:output:1)gru_42/while/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2 
gru_42/while/gru_cell_42/add_1?
"gru_42/while/gru_cell_42/Sigmoid_1Sigmoid"gru_42/while/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2$
"gru_42/while/gru_cell_42/Sigmoid_1?
gru_42/while/gru_cell_42/mulMul&gru_42/while/gru_cell_42/Sigmoid_1:y:0)gru_42/while/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_42/while/gru_cell_42/mul?
gru_42/while/gru_cell_42/add_2AddV2'gru_42/while/gru_cell_42/split:output:2 gru_42/while/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2 
gru_42/while/gru_cell_42/add_2?
gru_42/while/gru_cell_42/ReluRelu"gru_42/while/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_42/while/gru_cell_42/Relu?
gru_42/while/gru_cell_42/mul_1Mul$gru_42/while/gru_cell_42/Sigmoid:y:0gru_42_while_placeholder_2*
T0*(
_output_shapes
:??????????2 
gru_42/while/gru_cell_42/mul_1?
gru_42/while/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_42/while/gru_cell_42/sub/x?
gru_42/while/gru_cell_42/subSub'gru_42/while/gru_cell_42/sub/x:output:0$gru_42/while/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_42/while/gru_cell_42/sub?
gru_42/while/gru_cell_42/mul_2Mul gru_42/while/gru_cell_42/sub:z:0+gru_42/while/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
gru_42/while/gru_cell_42/mul_2?
gru_42/while/gru_cell_42/add_3AddV2"gru_42/while/gru_cell_42/mul_1:z:0"gru_42/while/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2 
gru_42/while/gru_cell_42/add_3?
1gru_42/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_42_while_placeholder_1gru_42_while_placeholder"gru_42/while/gru_cell_42/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_42/while/TensorArrayV2Write/TensorListSetItemj
gru_42/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_42/while/add/y?
gru_42/while/addAddV2gru_42_while_placeholdergru_42/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_42/while/addn
gru_42/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_42/while/add_1/y?
gru_42/while/add_1AddV2&gru_42_while_gru_42_while_loop_countergru_42/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_42/while/add_1?
gru_42/while/IdentityIdentitygru_42/while/add_1:z:0^gru_42/while/NoOp*
T0*
_output_shapes
: 2
gru_42/while/Identity?
gru_42/while/Identity_1Identity,gru_42_while_gru_42_while_maximum_iterations^gru_42/while/NoOp*
T0*
_output_shapes
: 2
gru_42/while/Identity_1?
gru_42/while/Identity_2Identitygru_42/while/add:z:0^gru_42/while/NoOp*
T0*
_output_shapes
: 2
gru_42/while/Identity_2?
gru_42/while/Identity_3IdentityAgru_42/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_42/while/NoOp*
T0*
_output_shapes
: 2
gru_42/while/Identity_3?
gru_42/while/Identity_4Identity"gru_42/while/gru_cell_42/add_3:z:0^gru_42/while/NoOp*
T0*(
_output_shapes
:??????????2
gru_42/while/Identity_4?
gru_42/while/NoOpNoOp/^gru_42/while/gru_cell_42/MatMul/ReadVariableOp1^gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp(^gru_42/while/gru_cell_42/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
gru_42/while/NoOp"L
#gru_42_while_gru_42_strided_slice_1%gru_42_while_gru_42_strided_slice_1_0"x
9gru_42_while_gru_cell_42_matmul_1_readvariableop_resource;gru_42_while_gru_cell_42_matmul_1_readvariableop_resource_0"t
7gru_42_while_gru_cell_42_matmul_readvariableop_resource9gru_42_while_gru_cell_42_matmul_readvariableop_resource_0"f
0gru_42_while_gru_cell_42_readvariableop_resource2gru_42_while_gru_cell_42_readvariableop_resource_0"7
gru_42_while_identitygru_42/while/Identity:output:0";
gru_42_while_identity_1 gru_42/while/Identity_1:output:0";
gru_42_while_identity_2 gru_42/while/Identity_2:output:0";
gru_42_while_identity_3 gru_42/while/Identity_3:output:0";
gru_42_while_identity_4 gru_42/while/Identity_4:output:0"?
_gru_42_while_tensorarrayv2read_tensorlistgetitem_gru_42_tensorarrayunstack_tensorlistfromtensoragru_42_while_tensorarrayv2read_tensorlistgetitem_gru_42_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2`
.gru_42/while/gru_cell_42/MatMul/ReadVariableOp.gru_42/while/gru_cell_42/MatMul/ReadVariableOp2d
0gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp0gru_42/while/gru_cell_42/MatMul_1/ReadVariableOp2R
'gru_42/while/gru_cell_42/ReadVariableOp'gru_42/while/gru_cell_42/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1573284
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1573284___redundant_placeholder05
1while_while_cond_1573284___redundant_placeholder15
1while_while_cond_1573284___redundant_placeholder25
1while_while_cond_1573284___redundant_placeholder3
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
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_1576684

inputs
states_0*
readvariableop_resource:	?1
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
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
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
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
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
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
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

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
-:?????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?
?
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_1571986

inputs

states*
readvariableop_resource:	?1
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
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
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
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
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu]
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:??????????2
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
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

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
-:?????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?E
?
while_body_1576159
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_43_readvariableop_resource_0:	?F
2while_gru_cell_43_matmul_readvariableop_resource_0:
??H
4while_gru_cell_43_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_43_readvariableop_resource:	?D
0while_gru_cell_43_matmul_readvariableop_resource:
??F
2while_gru_cell_43_matmul_1_readvariableop_resource:
????'while/gru_cell_43/MatMul/ReadVariableOp?)while/gru_cell_43/MatMul_1/ReadVariableOp? while/gru_cell_43/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_43/ReadVariableOpReadVariableOp+while_gru_cell_43_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_43/ReadVariableOp?
while/gru_cell_43/unstackUnpack(while/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_43/unstack?
'while/gru_cell_43/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_43_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_43/MatMul/ReadVariableOp?
while/gru_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul?
while/gru_cell_43/BiasAddBiasAdd"while/gru_cell_43/MatMul:product:0"while/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd?
!while/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_43/split/split_dim?
while/gru_cell_43/splitSplit*while/gru_cell_43/split/split_dim:output:0"while/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split?
)while/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_43_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_43/MatMul_1/ReadVariableOp?
while/gru_cell_43/MatMul_1MatMulwhile_placeholder_21while/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul_1?
while/gru_cell_43/BiasAdd_1BiasAdd$while/gru_cell_43/MatMul_1:product:0"while/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd_1?
while/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_43/Const?
#while/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_43/split_1/split_dim?
while/gru_cell_43/split_1SplitV$while/gru_cell_43/BiasAdd_1:output:0 while/gru_cell_43/Const:output:0,while/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split_1?
while/gru_cell_43/addAddV2 while/gru_cell_43/split:output:0"while/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add?
while/gru_cell_43/SigmoidSigmoidwhile/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid?
while/gru_cell_43/add_1AddV2 while/gru_cell_43/split:output:1"while/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_1?
while/gru_cell_43/Sigmoid_1Sigmoidwhile/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid_1?
while/gru_cell_43/mulMulwhile/gru_cell_43/Sigmoid_1:y:0"while/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul?
while/gru_cell_43/add_2AddV2 while/gru_cell_43/split:output:2while/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_2?
while/gru_cell_43/ReluReluwhile/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Relu?
while/gru_cell_43/mul_1Mulwhile/gru_cell_43/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_1w
while/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_43/sub/x?
while/gru_cell_43/subSub while/gru_cell_43/sub/x:output:0while/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/sub?
while/gru_cell_43/mul_2Mulwhile/gru_cell_43/sub:z:0$while/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_2?
while/gru_cell_43/add_3AddV2while/gru_cell_43/mul_1:z:0while/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_43/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_43/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_43/MatMul/ReadVariableOp*^while/gru_cell_43/MatMul_1/ReadVariableOp!^while/gru_cell_43/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_43_matmul_1_readvariableop_resource4while_gru_cell_43_matmul_1_readvariableop_resource_0"f
0while_gru_cell_43_matmul_readvariableop_resource2while_gru_cell_43_matmul_readvariableop_resource_0"X
)while_gru_cell_43_readvariableop_resource+while_gru_cell_43_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_43/MatMul/ReadVariableOp'while/gru_cell_43/MatMul/ReadVariableOp2V
)while/gru_cell_43/MatMul_1/ReadVariableOp)while/gru_cell_43/MatMul_1/ReadVariableOp2D
 while/gru_cell_43/ReadVariableOp while/gru_cell_43/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?0
?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1574081

inputs!
gru_42_1574047:	?!
gru_42_1574049:	?"
gru_42_1574051:
??!
gru_43_1574055:	?"
gru_43_1574057:
??"
gru_43_1574059:
??$
dense_42_1574063:
??
dense_42_1574065:	?$
dense_43_1574069:
??
dense_43_1574071:	?#
dense_44_1574075:	?
dense_44_1574077:
identity?? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall?"dropout_63/StatefulPartitionedCall?"dropout_64/StatefulPartitionedCall?"dropout_65/StatefulPartitionedCall?"dropout_66/StatefulPartitionedCall?gru_42/StatefulPartitionedCall?gru_43/StatefulPartitionedCall?
gru_42/StatefulPartitionedCallStatefulPartitionedCallinputsgru_42_1574047gru_42_1574049gru_42_1574051*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_42_layer_call_and_return_conditional_losses_15740042 
gru_42/StatefulPartitionedCall?
"dropout_63/StatefulPartitionedCallStatefulPartitionedCall'gru_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_63_layer_call_and_return_conditional_losses_15738352$
"dropout_63/StatefulPartitionedCall?
gru_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_63/StatefulPartitionedCall:output:0gru_43_1574055gru_43_1574057gru_43_1574059*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_43_layer_call_and_return_conditional_losses_15738062 
gru_43/StatefulPartitionedCall?
"dropout_64/StatefulPartitionedCallStatefulPartitionedCall'gru_43/StatefulPartitionedCall:output:0#^dropout_63/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_64_layer_call_and_return_conditional_losses_15736372$
"dropout_64/StatefulPartitionedCall?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall+dropout_64/StatefulPartitionedCall:output:0dense_42_1574063dense_42_1574065*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_42_layer_call_and_return_conditional_losses_15734202"
 dense_42/StatefulPartitionedCall?
"dropout_65/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0#^dropout_64/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_65_layer_call_and_return_conditional_losses_15736042$
"dropout_65/StatefulPartitionedCall?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_65/StatefulPartitionedCall:output:0dense_43_1574069dense_43_1574071*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_43_layer_call_and_return_conditional_losses_15734642"
 dense_43/StatefulPartitionedCall?
"dropout_66/StatefulPartitionedCallStatefulPartitionedCall)dense_43/StatefulPartitionedCall:output:0#^dropout_65/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_66_layer_call_and_return_conditional_losses_15735712$
"dropout_66/StatefulPartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall+dropout_66/StatefulPartitionedCall:output:0dense_44_1574075dense_44_1574077*
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
E__inference_dense_44_layer_call_and_return_conditional_losses_15735072"
 dense_44/StatefulPartitionedCall?
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall#^dropout_63/StatefulPartitionedCall#^dropout_64/StatefulPartitionedCall#^dropout_65/StatefulPartitionedCall#^dropout_66/StatefulPartitionedCall^gru_42/StatefulPartitionedCall^gru_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : : : 2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2H
"dropout_63/StatefulPartitionedCall"dropout_63/StatefulPartitionedCall2H
"dropout_64/StatefulPartitionedCall"dropout_64/StatefulPartitionedCall2H
"dropout_65/StatefulPartitionedCall"dropout_65/StatefulPartitionedCall2H
"dropout_66/StatefulPartitionedCall"dropout_66/StatefulPartitionedCall2@
gru_42/StatefulPartitionedCallgru_42/StatefulPartitionedCall2@
gru_43/StatefulPartitionedCallgru_43/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_1576723

inputs
states_0*
readvariableop_resource:	?1
matmul_readvariableop_resource:	?4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
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
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
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
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
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
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

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
-:?????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?E
?
while_body_1576006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_43_readvariableop_resource_0:	?F
2while_gru_cell_43_matmul_readvariableop_resource_0:
??H
4while_gru_cell_43_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_43_readvariableop_resource:	?D
0while_gru_cell_43_matmul_readvariableop_resource:
??F
2while_gru_cell_43_matmul_1_readvariableop_resource:
????'while/gru_cell_43/MatMul/ReadVariableOp?)while/gru_cell_43/MatMul_1/ReadVariableOp? while/gru_cell_43/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_43/ReadVariableOpReadVariableOp+while_gru_cell_43_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_43/ReadVariableOp?
while/gru_cell_43/unstackUnpack(while/gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_43/unstack?
'while/gru_cell_43/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_43_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02)
'while/gru_cell_43/MatMul/ReadVariableOp?
while/gru_cell_43/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul?
while/gru_cell_43/BiasAddBiasAdd"while/gru_cell_43/MatMul:product:0"while/gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd?
!while/gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_43/split/split_dim?
while/gru_cell_43/splitSplit*while/gru_cell_43/split/split_dim:output:0"while/gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split?
)while/gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_43_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_43/MatMul_1/ReadVariableOp?
while/gru_cell_43/MatMul_1MatMulwhile_placeholder_21while/gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/MatMul_1?
while/gru_cell_43/BiasAdd_1BiasAdd$while/gru_cell_43/MatMul_1:product:0"while/gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/BiasAdd_1?
while/gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_43/Const?
#while/gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_43/split_1/split_dim?
while/gru_cell_43/split_1SplitV$while/gru_cell_43/BiasAdd_1:output:0 while/gru_cell_43/Const:output:0,while/gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_43/split_1?
while/gru_cell_43/addAddV2 while/gru_cell_43/split:output:0"while/gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add?
while/gru_cell_43/SigmoidSigmoidwhile/gru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid?
while/gru_cell_43/add_1AddV2 while/gru_cell_43/split:output:1"while/gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_1?
while/gru_cell_43/Sigmoid_1Sigmoidwhile/gru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Sigmoid_1?
while/gru_cell_43/mulMulwhile/gru_cell_43/Sigmoid_1:y:0"while/gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul?
while/gru_cell_43/add_2AddV2 while/gru_cell_43/split:output:2while/gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_2?
while/gru_cell_43/ReluReluwhile/gru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/Relu?
while/gru_cell_43/mul_1Mulwhile/gru_cell_43/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_1w
while/gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_43/sub/x?
while/gru_cell_43/subSub while/gru_cell_43/sub/x:output:0while/gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/sub?
while/gru_cell_43/mul_2Mulwhile/gru_cell_43/sub:z:0$while/gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/mul_2?
while/gru_cell_43/add_3AddV2while/gru_cell_43/mul_1:z:0while/gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_43/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_43/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_43/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_43/MatMul/ReadVariableOp*^while/gru_cell_43/MatMul_1/ReadVariableOp!^while/gru_cell_43/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_43_matmul_1_readvariableop_resource4while_gru_cell_43_matmul_1_readvariableop_resource_0"f
0while_gru_cell_43_matmul_readvariableop_resource2while_gru_cell_43_matmul_readvariableop_resource_0"X
)while_gru_cell_43_readvariableop_resource+while_gru_cell_43_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_43/MatMul/ReadVariableOp'while/gru_cell_43/MatMul/ReadVariableOp2V
)while/gru_cell_43/MatMul_1/ReadVariableOp)while/gru_cell_43/MatMul_1/ReadVariableOp2D
 while/gru_cell_43/ReadVariableOp while/gru_cell_43/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?"
?
while_body_1572565
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_43_1572587_0:	?/
while_gru_cell_43_1572589_0:
??/
while_gru_cell_43_1572591_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_43_1572587:	?-
while_gru_cell_43_1572589:
??-
while_gru_cell_43_1572591:
????)while/gru_cell_43/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_43/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_43_1572587_0while_gru_cell_43_1572589_0while_gru_cell_43_1572591_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_15725522+
)while/gru_cell_43/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_43/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_43/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"8
while_gru_cell_43_1572587while_gru_cell_43_1572587_0"8
while_gru_cell_43_1572589while_gru_cell_43_1572589_0"8
while_gru_cell_43_1572591while_gru_cell_43_1572591_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2V
)while/gru_cell_43/StatefulPartitionedCall)while/gru_cell_43/StatefulPartitionedCall: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?X
?
C__inference_gru_42_layer_call_and_return_conditional_losses_1573207

inputs6
#gru_cell_42_readvariableop_resource:	?=
*gru_cell_42_matmul_readvariableop_resource:	?@
,gru_cell_42_matmul_1_readvariableop_resource:
??
identity??!gru_cell_42/MatMul/ReadVariableOp?#gru_cell_42/MatMul_1/ReadVariableOp?gru_cell_42/ReadVariableOp?whileD
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
B :?2
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
:??????????2
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
:?????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_42/ReadVariableOpReadVariableOp#gru_cell_42_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_42/ReadVariableOp?
gru_cell_42/unstackUnpack"gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_42/unstack?
!gru_cell_42/MatMul/ReadVariableOpReadVariableOp*gru_cell_42_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_42/MatMul/ReadVariableOp?
gru_cell_42/MatMulMatMulstrided_slice_2:output:0)gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul?
gru_cell_42/BiasAddBiasAddgru_cell_42/MatMul:product:0gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd?
gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split/split_dim?
gru_cell_42/splitSplit$gru_cell_42/split/split_dim:output:0gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split?
#gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_42_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_42/MatMul_1/ReadVariableOp?
gru_cell_42/MatMul_1MatMulzeros:output:0+gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/MatMul_1?
gru_cell_42/BiasAdd_1BiasAddgru_cell_42/MatMul_1:product:0gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/BiasAdd_1{
gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_42/Const?
gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_42/split_1/split_dim?
gru_cell_42/split_1SplitVgru_cell_42/BiasAdd_1:output:0gru_cell_42/Const:output:0&gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_42/split_1?
gru_cell_42/addAddV2gru_cell_42/split:output:0gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add}
gru_cell_42/SigmoidSigmoidgru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid?
gru_cell_42/add_1AddV2gru_cell_42/split:output:1gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_1?
gru_cell_42/Sigmoid_1Sigmoidgru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Sigmoid_1?
gru_cell_42/mulMulgru_cell_42/Sigmoid_1:y:0gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul?
gru_cell_42/add_2AddV2gru_cell_42/split:output:2gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_2v
gru_cell_42/ReluRelugru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/Relu?
gru_cell_42/mul_1Mulgru_cell_42/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_1k
gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_42/sub/x?
gru_cell_42/subSubgru_cell_42/sub/x:output:0gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/sub?
gru_cell_42/mul_2Mulgru_cell_42/sub:z:0gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/mul_2?
gru_cell_42/add_3AddV2gru_cell_42/mul_1:z:0gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_42/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_42_readvariableop_resource*gru_cell_42_matmul_readvariableop_resource,gru_cell_42_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1573118*
condR
while_cond_1573117*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
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
:??????????2
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
:??????????2

Identity?
NoOpNoOp"^gru_cell_42/MatMul/ReadVariableOp$^gru_cell_42/MatMul_1/ReadVariableOp^gru_cell_42/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2F
!gru_cell_42/MatMul/ReadVariableOp!gru_cell_42/MatMul/ReadVariableOp2J
#gru_cell_42/MatMul_1/ReadVariableOp#gru_cell_42/MatMul_1/ReadVariableOp28
gru_cell_42/ReadVariableOpgru_cell_42/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
gru_43_while_cond_1574461*
&gru_43_while_gru_43_while_loop_counter0
,gru_43_while_gru_43_while_maximum_iterations
gru_43_while_placeholder
gru_43_while_placeholder_1
gru_43_while_placeholder_2,
(gru_43_while_less_gru_43_strided_slice_1C
?gru_43_while_gru_43_while_cond_1574461___redundant_placeholder0C
?gru_43_while_gru_43_while_cond_1574461___redundant_placeholder1C
?gru_43_while_gru_43_while_cond_1574461___redundant_placeholder2C
?gru_43_while_gru_43_while_cond_1574461___redundant_placeholder3
gru_43_while_identity
?
gru_43/while/LessLessgru_43_while_placeholder(gru_43_while_less_gru_43_strided_slice_1*
T0*
_output_shapes
: 2
gru_43/while/Lessr
gru_43/while/IdentityIdentitygru_43/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_43/while/Identity"7
gru_43_while_identitygru_43/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :??????????: ::::: 
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
:??????????:

_output_shapes
: :

_output_shapes
:
?E
?
while_body_1575629
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_42_readvariableop_resource_0:	?E
2while_gru_cell_42_matmul_readvariableop_resource_0:	?H
4while_gru_cell_42_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_42_readvariableop_resource:	?C
0while_gru_cell_42_matmul_readvariableop_resource:	?F
2while_gru_cell_42_matmul_1_readvariableop_resource:
????'while/gru_cell_42/MatMul/ReadVariableOp?)while/gru_cell_42/MatMul_1/ReadVariableOp? while/gru_cell_42/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_42/ReadVariableOpReadVariableOp+while_gru_cell_42_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_42/ReadVariableOp?
while/gru_cell_42/unstackUnpack(while/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_42/unstack?
'while/gru_cell_42/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_42/MatMul/ReadVariableOp?
while/gru_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul?
while/gru_cell_42/BiasAddBiasAdd"while/gru_cell_42/MatMul:product:0"while/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd?
!while/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_42/split/split_dim?
while/gru_cell_42/splitSplit*while/gru_cell_42/split/split_dim:output:0"while/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split?
)while/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_42_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_42/MatMul_1/ReadVariableOp?
while/gru_cell_42/MatMul_1MatMulwhile_placeholder_21while/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul_1?
while/gru_cell_42/BiasAdd_1BiasAdd$while/gru_cell_42/MatMul_1:product:0"while/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd_1?
while/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_42/Const?
#while/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_42/split_1/split_dim?
while/gru_cell_42/split_1SplitV$while/gru_cell_42/BiasAdd_1:output:0 while/gru_cell_42/Const:output:0,while/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split_1?
while/gru_cell_42/addAddV2 while/gru_cell_42/split:output:0"while/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add?
while/gru_cell_42/SigmoidSigmoidwhile/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid?
while/gru_cell_42/add_1AddV2 while/gru_cell_42/split:output:1"while/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_1?
while/gru_cell_42/Sigmoid_1Sigmoidwhile/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid_1?
while/gru_cell_42/mulMulwhile/gru_cell_42/Sigmoid_1:y:0"while/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul?
while/gru_cell_42/add_2AddV2 while/gru_cell_42/split:output:2while/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_2?
while/gru_cell_42/ReluReluwhile/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Relu?
while/gru_cell_42/mul_1Mulwhile/gru_cell_42/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_1w
while/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_42/sub/x?
while/gru_cell_42/subSub while/gru_cell_42/sub/x:output:0while/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/sub?
while/gru_cell_42/mul_2Mulwhile/gru_cell_42/sub:z:0$while/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_2?
while/gru_cell_42/add_3AddV2while/gru_cell_42/mul_1:z:0while/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_42/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_42/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_42/MatMul/ReadVariableOp*^while/gru_cell_42/MatMul_1/ReadVariableOp!^while/gru_cell_42/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_42_matmul_1_readvariableop_resource4while_gru_cell_42_matmul_1_readvariableop_resource_0"f
0while_gru_cell_42_matmul_readvariableop_resource2while_gru_cell_42_matmul_readvariableop_resource_0"X
)while_gru_cell_42_readvariableop_resource+while_gru_cell_42_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_42/MatMul/ReadVariableOp'while/gru_cell_42/MatMul/ReadVariableOp2V
)while/gru_cell_42/MatMul_1/ReadVariableOp)while/gru_cell_42/MatMul_1/ReadVariableOp2D
 while/gru_cell_42/ReadVariableOp while/gru_cell_42/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
? 
?
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_1576829

inputs
states_0*
readvariableop_resource:	?2
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??
identity

identity_1??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
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
<:??????????:??????????:??????????*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1c
ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
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
<:??????????:??????????:??????????*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:??????????2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:??????????2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:??????????2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:??????????2
add_2R
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:??????????2
Relu_
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:??????????2
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
:??????????2
sube
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:??????????2
add_3e
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:??????????2

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
.:??????????:??????????: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0
?"
?
while_body_1572758
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_43_1572780_0:	?/
while_gru_cell_43_1572782_0:
??/
while_gru_cell_43_1572784_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_43_1572780:	?-
while_gru_cell_43_1572782:
??-
while_gru_cell_43_1572784:
????)while/gru_cell_43/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_43/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_43_1572780_0while_gru_cell_43_1572782_0while_gru_cell_43_1572784_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_15726952+
)while/gru_cell_43/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_43/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/gru_cell_43/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp*^while/gru_cell_43/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"8
while_gru_cell_43_1572780while_gru_cell_43_1572780_0"8
while_gru_cell_43_1572782while_gru_cell_43_1572782_0"8
while_gru_cell_43_1572784while_gru_cell_43_1572784_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2V
)while/gru_cell_43/StatefulPartitionedCall)while/gru_cell_43/StatefulPartitionedCall: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?X
?
C__inference_gru_43_layer_call_and_return_conditional_losses_1576401

inputs6
#gru_cell_43_readvariableop_resource:	?>
*gru_cell_43_matmul_readvariableop_resource:
??@
,gru_cell_43_matmul_1_readvariableop_resource:
??
identity??!gru_cell_43/MatMul/ReadVariableOp?#gru_cell_43/MatMul_1/ReadVariableOp?gru_cell_43/ReadVariableOp?whileD
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
B :?2
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
:??????????2
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
:??????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_43/ReadVariableOpReadVariableOp#gru_cell_43_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_43/ReadVariableOp?
gru_cell_43/unstackUnpack"gru_cell_43/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_43/unstack?
!gru_cell_43/MatMul/ReadVariableOpReadVariableOp*gru_cell_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!gru_cell_43/MatMul/ReadVariableOp?
gru_cell_43/MatMulMatMulstrided_slice_2:output:0)gru_cell_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul?
gru_cell_43/BiasAddBiasAddgru_cell_43/MatMul:product:0gru_cell_43/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd?
gru_cell_43/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split/split_dim?
gru_cell_43/splitSplit$gru_cell_43/split/split_dim:output:0gru_cell_43/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split?
#gru_cell_43/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_43_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#gru_cell_43/MatMul_1/ReadVariableOp?
gru_cell_43/MatMul_1MatMulzeros:output:0+gru_cell_43/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/MatMul_1?
gru_cell_43/BiasAdd_1BiasAddgru_cell_43/MatMul_1:product:0gru_cell_43/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/BiasAdd_1{
gru_cell_43/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
gru_cell_43/Const?
gru_cell_43/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_43/split_1/split_dim?
gru_cell_43/split_1SplitVgru_cell_43/BiasAdd_1:output:0gru_cell_43/Const:output:0&gru_cell_43/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
gru_cell_43/split_1?
gru_cell_43/addAddV2gru_cell_43/split:output:0gru_cell_43/split_1:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add}
gru_cell_43/SigmoidSigmoidgru_cell_43/add:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid?
gru_cell_43/add_1AddV2gru_cell_43/split:output:1gru_cell_43/split_1:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_1?
gru_cell_43/Sigmoid_1Sigmoidgru_cell_43/add_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Sigmoid_1?
gru_cell_43/mulMulgru_cell_43/Sigmoid_1:y:0gru_cell_43/split_1:output:2*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul?
gru_cell_43/add_2AddV2gru_cell_43/split:output:2gru_cell_43/mul:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_2v
gru_cell_43/ReluRelugru_cell_43/add_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/Relu?
gru_cell_43/mul_1Mulgru_cell_43/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_1k
gru_cell_43/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_43/sub/x?
gru_cell_43/subSubgru_cell_43/sub/x:output:0gru_cell_43/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/sub?
gru_cell_43/mul_2Mulgru_cell_43/sub:z:0gru_cell_43/Relu:activations:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/mul_2?
gru_cell_43/add_3AddV2gru_cell_43/mul_1:z:0gru_cell_43/mul_2:z:0*
T0*(
_output_shapes
:??????????2
gru_cell_43/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_43_readvariableop_resource*gru_cell_43_matmul_readvariableop_resource,gru_cell_43_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :??????????: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1576312*
condR
while_cond_1576311*9
output_shapes(
&: : : : :??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
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
:??????????*
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
:??????????2
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
:??????????2

Identity?
NoOpNoOp"^gru_cell_43/MatMul/ReadVariableOp$^gru_cell_43/MatMul_1/ReadVariableOp^gru_cell_43/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2F
!gru_cell_43/MatMul/ReadVariableOp!gru_cell_43/MatMul/ReadVariableOp2J
#gru_cell_43/MatMul_1/ReadVariableOp#gru_cell_43/MatMul_1/ReadVariableOp28
gru_cell_43/ReadVariableOpgru_cell_43/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?E
?
while_body_1573118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_42_readvariableop_resource_0:	?E
2while_gru_cell_42_matmul_readvariableop_resource_0:	?H
4while_gru_cell_42_matmul_1_readvariableop_resource_0:
??
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_42_readvariableop_resource:	?C
0while_gru_cell_42_matmul_readvariableop_resource:	?F
2while_gru_cell_42_matmul_1_readvariableop_resource:
????'while/gru_cell_42/MatMul/ReadVariableOp?)while/gru_cell_42/MatMul_1/ReadVariableOp? while/gru_cell_42/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_42/ReadVariableOpReadVariableOp+while_gru_cell_42_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_42/ReadVariableOp?
while/gru_cell_42/unstackUnpack(while/gru_cell_42/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_42/unstack?
'while/gru_cell_42/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_42_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_42/MatMul/ReadVariableOp?
while/gru_cell_42/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul?
while/gru_cell_42/BiasAddBiasAdd"while/gru_cell_42/MatMul:product:0"while/gru_cell_42/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd?
!while/gru_cell_42/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_42/split/split_dim?
while/gru_cell_42/splitSplit*while/gru_cell_42/split/split_dim:output:0"while/gru_cell_42/BiasAdd:output:0*
T0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split?
)while/gru_cell_42/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_42_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)while/gru_cell_42/MatMul_1/ReadVariableOp?
while/gru_cell_42/MatMul_1MatMulwhile_placeholder_21while/gru_cell_42/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/MatMul_1?
while/gru_cell_42/BiasAdd_1BiasAdd$while/gru_cell_42/MatMul_1:product:0"while/gru_cell_42/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/BiasAdd_1?
while/gru_cell_42/ConstConst*
_output_shapes
:*
dtype0*!
valueB"?   ?   ????2
while/gru_cell_42/Const?
#while/gru_cell_42/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_42/split_1/split_dim?
while/gru_cell_42/split_1SplitV$while/gru_cell_42/BiasAdd_1:output:0 while/gru_cell_42/Const:output:0,while/gru_cell_42/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:??????????:??????????:??????????*
	num_split2
while/gru_cell_42/split_1?
while/gru_cell_42/addAddV2 while/gru_cell_42/split:output:0"while/gru_cell_42/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add?
while/gru_cell_42/SigmoidSigmoidwhile/gru_cell_42/add:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid?
while/gru_cell_42/add_1AddV2 while/gru_cell_42/split:output:1"while/gru_cell_42/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_1?
while/gru_cell_42/Sigmoid_1Sigmoidwhile/gru_cell_42/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Sigmoid_1?
while/gru_cell_42/mulMulwhile/gru_cell_42/Sigmoid_1:y:0"while/gru_cell_42/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul?
while/gru_cell_42/add_2AddV2 while/gru_cell_42/split:output:2while/gru_cell_42/mul:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_2?
while/gru_cell_42/ReluReluwhile/gru_cell_42/add_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/Relu?
while/gru_cell_42/mul_1Mulwhile/gru_cell_42/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_1w
while/gru_cell_42/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_42/sub/x?
while/gru_cell_42/subSub while/gru_cell_42/sub/x:output:0while/gru_cell_42/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/sub?
while/gru_cell_42/mul_2Mulwhile/gru_cell_42/sub:z:0$while/gru_cell_42/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/mul_2?
while/gru_cell_42/add_3AddV2while/gru_cell_42/mul_1:z:0while/gru_cell_42/mul_2:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_42/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_42/add_3:z:0*
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
while/Identity_4Identitywhile/gru_cell_42/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?

while/NoOpNoOp(^while/gru_cell_42/MatMul/ReadVariableOp*^while/gru_cell_42/MatMul_1/ReadVariableOp!^while/gru_cell_42/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp"j
2while_gru_cell_42_matmul_1_readvariableop_resource4while_gru_cell_42_matmul_1_readvariableop_resource_0"f
0while_gru_cell_42_matmul_readvariableop_resource2while_gru_cell_42_matmul_readvariableop_resource_0"X
)while_gru_cell_42_readvariableop_resource+while_gru_cell_42_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :??????????: : : : : 2R
'while/gru_cell_42/MatMul/ReadVariableOp'while/gru_cell_42/MatMul/ReadVariableOp2V
)while/gru_cell_42/MatMul_1/ReadVariableOp)while/gru_cell_42/MatMul_1/ReadVariableOp2D
 while/gru_cell_42/ReadVariableOp while/gru_cell_42/ReadVariableOp: 
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
:??????????:

_output_shapes
: :

_output_shapes
: 
?!
?
E__inference_dense_43_layer_call_and_return_conditional_losses_1573464

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
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
:??????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1574248
gru_42_input
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:
??
	unknown_4:
??
	unknown_5:
??
	unknown_6:	?
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_42_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_15719162
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
/:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_42_input
? 
?
E__inference_dense_44_layer_call_and_return_conditional_losses_1573507

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
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
:??????????2
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_63_layer_call_fn_1575784

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_63_layer_call_and_return_conditional_losses_15732202
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_66_layer_call_and_return_conditional_losses_1576584

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?a
?
 __inference__traced_save_1577015
file_prefix.
*savev2_dense_42_kernel_read_readvariableop,
(savev2_dense_42_bias_read_readvariableop.
*savev2_dense_43_kernel_read_readvariableop,
(savev2_dense_43_bias_read_readvariableop.
*savev2_dense_44_kernel_read_readvariableop,
(savev2_dense_44_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_gru_42_gru_cell_42_kernel_read_readvariableopB
>savev2_gru_42_gru_cell_42_recurrent_kernel_read_readvariableop6
2savev2_gru_42_gru_cell_42_bias_read_readvariableop8
4savev2_gru_43_gru_cell_43_kernel_read_readvariableopB
>savev2_gru_43_gru_cell_43_recurrent_kernel_read_readvariableop6
2savev2_gru_43_gru_cell_43_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_42_kernel_m_read_readvariableop3
/savev2_adam_dense_42_bias_m_read_readvariableop5
1savev2_adam_dense_43_kernel_m_read_readvariableop3
/savev2_adam_dense_43_bias_m_read_readvariableop5
1savev2_adam_dense_44_kernel_m_read_readvariableop3
/savev2_adam_dense_44_bias_m_read_readvariableop?
;savev2_adam_gru_42_gru_cell_42_kernel_m_read_readvariableopI
Esavev2_adam_gru_42_gru_cell_42_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_42_gru_cell_42_bias_m_read_readvariableop?
;savev2_adam_gru_43_gru_cell_43_kernel_m_read_readvariableopI
Esavev2_adam_gru_43_gru_cell_43_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_43_gru_cell_43_bias_m_read_readvariableop5
1savev2_adam_dense_42_kernel_v_read_readvariableop3
/savev2_adam_dense_42_bias_v_read_readvariableop5
1savev2_adam_dense_43_kernel_v_read_readvariableop3
/savev2_adam_dense_43_bias_v_read_readvariableop5
1savev2_adam_dense_44_kernel_v_read_readvariableop3
/savev2_adam_dense_44_bias_v_read_readvariableop?
;savev2_adam_gru_42_gru_cell_42_kernel_v_read_readvariableopI
Esavev2_adam_gru_42_gru_cell_42_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_42_gru_cell_42_bias_v_read_readvariableop?
;savev2_adam_gru_43_gru_cell_43_kernel_v_read_readvariableopI
Esavev2_adam_gru_43_gru_cell_43_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_43_gru_cell_43_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_42_kernel_read_readvariableop(savev2_dense_42_bias_read_readvariableop*savev2_dense_43_kernel_read_readvariableop(savev2_dense_43_bias_read_readvariableop*savev2_dense_44_kernel_read_readvariableop(savev2_dense_44_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_gru_42_gru_cell_42_kernel_read_readvariableop>savev2_gru_42_gru_cell_42_recurrent_kernel_read_readvariableop2savev2_gru_42_gru_cell_42_bias_read_readvariableop4savev2_gru_43_gru_cell_43_kernel_read_readvariableop>savev2_gru_43_gru_cell_43_recurrent_kernel_read_readvariableop2savev2_gru_43_gru_cell_43_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_42_kernel_m_read_readvariableop/savev2_adam_dense_42_bias_m_read_readvariableop1savev2_adam_dense_43_kernel_m_read_readvariableop/savev2_adam_dense_43_bias_m_read_readvariableop1savev2_adam_dense_44_kernel_m_read_readvariableop/savev2_adam_dense_44_bias_m_read_readvariableop;savev2_adam_gru_42_gru_cell_42_kernel_m_read_readvariableopEsavev2_adam_gru_42_gru_cell_42_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_42_gru_cell_42_bias_m_read_readvariableop;savev2_adam_gru_43_gru_cell_43_kernel_m_read_readvariableopEsavev2_adam_gru_43_gru_cell_43_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_43_gru_cell_43_bias_m_read_readvariableop1savev2_adam_dense_42_kernel_v_read_readvariableop/savev2_adam_dense_42_bias_v_read_readvariableop1savev2_adam_dense_43_kernel_v_read_readvariableop/savev2_adam_dense_43_bias_v_read_readvariableop1savev2_adam_dense_44_kernel_v_read_readvariableop/savev2_adam_dense_44_bias_v_read_readvariableop;savev2_adam_gru_42_gru_cell_42_kernel_v_read_readvariableopEsavev2_adam_gru_42_gru_cell_42_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_42_gru_cell_42_bias_v_read_readvariableop;savev2_adam_gru_43_gru_cell_43_kernel_v_read_readvariableopEsavev2_adam_gru_43_gru_cell_43_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_43_gru_cell_43_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
??:?:
??:?:	?:: : : : : :	?:
??:	?:
??:
??:	?: : : : :
??:?:
??:?:	?::	?:
??:	?:
??:
??:	?:
??:?:
??:?:	?::	?:
??:	?:
??:
??:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 
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
:	?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:
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
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:&"
 
_output_shapes
:
??:& "
 
_output_shapes
:
??:%!!

_output_shapes
:	?:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?: '

_output_shapes
::%(!

_output_shapes
:	?:&)"
 
_output_shapes
:
??:%*!

_output_shapes
:	?:&+"
 
_output_shapes
:
??:&,"
 
_output_shapes
:
??:%-!

_output_shapes
:	?:.

_output_shapes
: 
?
e
G__inference_dropout_65_layer_call_and_return_conditional_losses_1573431

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
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
gru_42_input9
serving_default_gru_42_input:0?????????@
dense_444
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
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"
_tf_keras_sequential
?
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_rnn_layer
?
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_rnn_layer
?
 trainable_variables
!	variables
"regularization_losses
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
*trainable_variables
+	variables
,regularization_losses
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
4trainable_variables
5	variables
6regularization_losses
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
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
trainable_variables
Ilayer_regularization_losses
Jnon_trainable_variables
Klayer_metrics

Llayers
Mmetrics
	variables
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
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
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
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
Tlayer_metrics

Ulayers
Vmetrics
	variables
regularization_losses

Wstates
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
trainable_variables
Xlayer_regularization_losses
Ynon_trainable_variables
Zlayer_metrics

[layers
\metrics
	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Fkernel
Grecurrent_kernel
Hbias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
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
trainable_variables
alayer_regularization_losses
bnon_trainable_variables
clayer_metrics

dlayers
emetrics
	variables
regularization_losses

fstates
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
 trainable_variables
glayer_regularization_losses
hnon_trainable_variables
ilayer_metrics

jlayers
kmetrics
!	variables
"regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_42/kernel
:?2dense_42/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
&trainable_variables
llayer_regularization_losses
mnon_trainable_variables
nlayer_metrics

olayers
pmetrics
'	variables
(regularization_losses
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
*trainable_variables
qlayer_regularization_losses
rnon_trainable_variables
slayer_metrics

tlayers
umetrics
+	variables
,regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_43/kernel
:?2dense_43/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0trainable_variables
vlayer_regularization_losses
wnon_trainable_variables
xlayer_metrics

ylayers
zmetrics
1	variables
2regularization_losses
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
4trainable_variables
{layer_regularization_losses
|non_trainable_variables
}layer_metrics

~layers
metrics
5	variables
6regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_44/kernel
:2dense_44/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
:trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
?metrics
;	variables
<regularization_losses
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
,:*	?2gru_42/gru_cell_42/kernel
7:5
??2#gru_42/gru_cell_42/recurrent_kernel
*:(	?2gru_42/gru_cell_42/bias
-:+
??2gru_43/gru_cell_43/kernel
7:5
??2#gru_43/gru_cell_43/recurrent_kernel
*:(	?2gru_43/gru_cell_43/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
0
?0
?1"
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
Ntrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
?metrics
O	variables
Pregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
]trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?layers
?metrics
^	variables
_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
??2Adam/dense_42/kernel/m
!:?2Adam/dense_42/bias/m
(:&
??2Adam/dense_43/kernel/m
!:?2Adam/dense_43/bias/m
':%	?2Adam/dense_44/kernel/m
 :2Adam/dense_44/bias/m
1:/	?2 Adam/gru_42/gru_cell_42/kernel/m
<::
??2*Adam/gru_42/gru_cell_42/recurrent_kernel/m
/:-	?2Adam/gru_42/gru_cell_42/bias/m
2:0
??2 Adam/gru_43/gru_cell_43/kernel/m
<::
??2*Adam/gru_43/gru_cell_43/recurrent_kernel/m
/:-	?2Adam/gru_43/gru_cell_43/bias/m
(:&
??2Adam/dense_42/kernel/v
!:?2Adam/dense_42/bias/v
(:&
??2Adam/dense_43/kernel/v
!:?2Adam/dense_43/bias/v
':%	?2Adam/dense_44/kernel/v
 :2Adam/dense_44/bias/v
1:/	?2 Adam/gru_42/gru_cell_42/kernel/v
<::
??2*Adam/gru_42/gru_cell_42/recurrent_kernel/v
/:-	?2Adam/gru_42/gru_cell_42/bias/v
2:0
??2 Adam/gru_43/gru_cell_43/kernel/v
<::
??2*Adam/gru_43/gru_cell_43/recurrent_kernel/v
/:-	?2Adam/gru_43/gru_cell_43/bias/v
?2?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1574634
J__inference_sequential_21_layer_call_and_return_conditional_losses_1575048
J__inference_sequential_21_layer_call_and_return_conditional_losses_1574174
J__inference_sequential_21_layer_call_and_return_conditional_losses_1574211?
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
/__inference_sequential_21_layer_call_fn_1573541
/__inference_sequential_21_layer_call_fn_1575077
/__inference_sequential_21_layer_call_fn_1575106
/__inference_sequential_21_layer_call_fn_1574137?
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
"__inference__wrapped_model_1571916gru_42_input"?
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
C__inference_gru_42_layer_call_and_return_conditional_losses_1575259
C__inference_gru_42_layer_call_and_return_conditional_losses_1575412
C__inference_gru_42_layer_call_and_return_conditional_losses_1575565
C__inference_gru_42_layer_call_and_return_conditional_losses_1575718?
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
(__inference_gru_42_layer_call_fn_1575729
(__inference_gru_42_layer_call_fn_1575740
(__inference_gru_42_layer_call_fn_1575751
(__inference_gru_42_layer_call_fn_1575762?
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
G__inference_dropout_63_layer_call_and_return_conditional_losses_1575767
G__inference_dropout_63_layer_call_and_return_conditional_losses_1575779?
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
,__inference_dropout_63_layer_call_fn_1575784
,__inference_dropout_63_layer_call_fn_1575789?
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
C__inference_gru_43_layer_call_and_return_conditional_losses_1575942
C__inference_gru_43_layer_call_and_return_conditional_losses_1576095
C__inference_gru_43_layer_call_and_return_conditional_losses_1576248
C__inference_gru_43_layer_call_and_return_conditional_losses_1576401?
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
(__inference_gru_43_layer_call_fn_1576412
(__inference_gru_43_layer_call_fn_1576423
(__inference_gru_43_layer_call_fn_1576434
(__inference_gru_43_layer_call_fn_1576445?
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
G__inference_dropout_64_layer_call_and_return_conditional_losses_1576450
G__inference_dropout_64_layer_call_and_return_conditional_losses_1576462?
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
,__inference_dropout_64_layer_call_fn_1576467
,__inference_dropout_64_layer_call_fn_1576472?
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
E__inference_dense_42_layer_call_and_return_conditional_losses_1576503?
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
*__inference_dense_42_layer_call_fn_1576512?
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
G__inference_dropout_65_layer_call_and_return_conditional_losses_1576517
G__inference_dropout_65_layer_call_and_return_conditional_losses_1576529?
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
,__inference_dropout_65_layer_call_fn_1576534
,__inference_dropout_65_layer_call_fn_1576539?
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
E__inference_dense_43_layer_call_and_return_conditional_losses_1576570?
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
*__inference_dense_43_layer_call_fn_1576579?
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
G__inference_dropout_66_layer_call_and_return_conditional_losses_1576584
G__inference_dropout_66_layer_call_and_return_conditional_losses_1576596?
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
,__inference_dropout_66_layer_call_fn_1576601
,__inference_dropout_66_layer_call_fn_1576606?
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
E__inference_dense_44_layer_call_and_return_conditional_losses_1576636?
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
*__inference_dense_44_layer_call_fn_1576645?
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
%__inference_signature_wrapper_1574248gru_42_input"?
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
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_1576684
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_1576723?
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
-__inference_gru_cell_42_layer_call_fn_1576737
-__inference_gru_cell_42_layer_call_fn_1576751?
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
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_1576790
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_1576829?
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
-__inference_gru_cell_43_layer_call_fn_1576843
-__inference_gru_cell_43_layer_call_fn_1576857?
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
"__inference__wrapped_model_1571916?ECDHFG$%./899?6
/?,
*?'
gru_42_input?????????
? "7?4
2
dense_44&?#
dense_44??????????
E__inference_dense_42_layer_call_and_return_conditional_losses_1576503f$%4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
*__inference_dense_42_layer_call_fn_1576512Y$%4?1
*?'
%?"
inputs??????????
? "????????????
E__inference_dense_43_layer_call_and_return_conditional_losses_1576570f./4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????
? ?
*__inference_dense_43_layer_call_fn_1576579Y./4?1
*?'
%?"
inputs??????????
? "????????????
E__inference_dense_44_layer_call_and_return_conditional_losses_1576636e894?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
*__inference_dense_44_layer_call_fn_1576645X894?1
*?'
%?"
inputs??????????
? "???????????
G__inference_dropout_63_layer_call_and_return_conditional_losses_1575767f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_63_layer_call_and_return_conditional_losses_1575779f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_63_layer_call_fn_1575784Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_63_layer_call_fn_1575789Y8?5
.?+
%?"
inputs??????????
p
? "????????????
G__inference_dropout_64_layer_call_and_return_conditional_losses_1576450f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_64_layer_call_and_return_conditional_losses_1576462f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_64_layer_call_fn_1576467Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_64_layer_call_fn_1576472Y8?5
.?+
%?"
inputs??????????
p
? "????????????
G__inference_dropout_65_layer_call_and_return_conditional_losses_1576517f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_65_layer_call_and_return_conditional_losses_1576529f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_65_layer_call_fn_1576534Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_65_layer_call_fn_1576539Y8?5
.?+
%?"
inputs??????????
p
? "????????????
G__inference_dropout_66_layer_call_and_return_conditional_losses_1576584f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
G__inference_dropout_66_layer_call_and_return_conditional_losses_1576596f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
,__inference_dropout_66_layer_call_fn_1576601Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
,__inference_dropout_66_layer_call_fn_1576606Y8?5
.?+
%?"
inputs??????????
p
? "????????????
C__inference_gru_42_layer_call_and_return_conditional_losses_1575259?ECDO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
C__inference_gru_42_layer_call_and_return_conditional_losses_1575412?ECDO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
C__inference_gru_42_layer_call_and_return_conditional_losses_1575565rECD??<
5?2
$?!
inputs?????????

 
p 

 
? "*?'
 ?
0??????????
? ?
C__inference_gru_42_layer_call_and_return_conditional_losses_1575718rECD??<
5?2
$?!
inputs?????????

 
p

 
? "*?'
 ?
0??????????
? ?
(__inference_gru_42_layer_call_fn_1575729~ECDO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "&?#????????????????????
(__inference_gru_42_layer_call_fn_1575740~ECDO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "&?#????????????????????
(__inference_gru_42_layer_call_fn_1575751eECD??<
5?2
$?!
inputs?????????

 
p 

 
? "????????????
(__inference_gru_42_layer_call_fn_1575762eECD??<
5?2
$?!
inputs?????????

 
p

 
? "????????????
C__inference_gru_43_layer_call_and_return_conditional_losses_1575942?HFGP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
C__inference_gru_43_layer_call_and_return_conditional_losses_1576095?HFGP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
C__inference_gru_43_layer_call_and_return_conditional_losses_1576248sHFG@?=
6?3
%?"
inputs??????????

 
p 

 
? "*?'
 ?
0??????????
? ?
C__inference_gru_43_layer_call_and_return_conditional_losses_1576401sHFG@?=
6?3
%?"
inputs??????????

 
p

 
? "*?'
 ?
0??????????
? ?
(__inference_gru_43_layer_call_fn_1576412HFGP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#????????????????????
(__inference_gru_43_layer_call_fn_1576423HFGP?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#????????????????????
(__inference_gru_43_layer_call_fn_1576434fHFG@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
(__inference_gru_43_layer_call_fn_1576445fHFG@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_1576684?ECD]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p 
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
H__inference_gru_cell_42_layer_call_and_return_conditional_losses_1576723?ECD]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
-__inference_gru_cell_42_layer_call_fn_1576737?ECD]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p 
? "F?C
?
0??????????
#? 
?
1/0???????????
-__inference_gru_cell_42_layer_call_fn_1576751?ECD]?Z
S?P
 ?
inputs?????????
(?%
#? 
states/0??????????
p
? "F?C
?
0??????????
#? 
?
1/0???????????
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_1576790?HFG^?[
T?Q
!?
inputs??????????
(?%
#? 
states/0??????????
p 
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
H__inference_gru_cell_43_layer_call_and_return_conditional_losses_1576829?HFG^?[
T?Q
!?
inputs??????????
(?%
#? 
states/0??????????
p
? "T?Q
J?G
?
0/0??????????
%?"
 ?
0/1/0??????????
? ?
-__inference_gru_cell_43_layer_call_fn_1576843?HFG^?[
T?Q
!?
inputs??????????
(?%
#? 
states/0??????????
p 
? "F?C
?
0??????????
#? 
?
1/0???????????
-__inference_gru_cell_43_layer_call_fn_1576857?HFG^?[
T?Q
!?
inputs??????????
(?%
#? 
states/0??????????
p
? "F?C
?
0??????????
#? 
?
1/0???????????
J__inference_sequential_21_layer_call_and_return_conditional_losses_1574174|ECDHFG$%./89A?>
7?4
*?'
gru_42_input?????????
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1574211|ECDHFG$%./89A?>
7?4
*?'
gru_42_input?????????
p

 
? ")?&
?
0?????????
? ?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1574634vECDHFG$%./89;?8
1?.
$?!
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_21_layer_call_and_return_conditional_losses_1575048vECDHFG$%./89;?8
1?.
$?!
inputs?????????
p

 
? ")?&
?
0?????????
? ?
/__inference_sequential_21_layer_call_fn_1573541oECDHFG$%./89A?>
7?4
*?'
gru_42_input?????????
p 

 
? "???????????
/__inference_sequential_21_layer_call_fn_1574137oECDHFG$%./89A?>
7?4
*?'
gru_42_input?????????
p

 
? "???????????
/__inference_sequential_21_layer_call_fn_1575077iECDHFG$%./89;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
/__inference_sequential_21_layer_call_fn_1575106iECDHFG$%./89;?8
1?.
$?!
inputs?????????
p

 
? "???????????
%__inference_signature_wrapper_1574248?ECDHFG$%./89I?F
? 
??<
:
gru_42_input*?'
gru_42_input?????????"7?4
2
dense_44&?#
dense_44?????????