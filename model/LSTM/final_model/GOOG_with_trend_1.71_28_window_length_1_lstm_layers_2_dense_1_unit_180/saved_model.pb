í:
¦
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
­
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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
«
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements#
handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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
"serve*2.6.02unknown8¼®8
|
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´´* 
shared_namedense_31/kernel
u
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel* 
_output_shapes
:
´´*
dtype0
s
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:´*
shared_namedense_31/bias
l
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes	
:´*
dtype0
{
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	´* 
shared_namedense_32/kernel
t
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes
:	´*
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
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

lstm_29/lstm_cell_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ð*,
shared_namelstm_29/lstm_cell_29/kernel

/lstm_29/lstm_cell_29/kernel/Read/ReadVariableOpReadVariableOplstm_29/lstm_cell_29/kernel*
_output_shapes
:	Ð*
dtype0
¨
%lstm_29/lstm_cell_29/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*6
shared_name'%lstm_29/lstm_cell_29/recurrent_kernel
¡
9lstm_29/lstm_cell_29/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_29/lstm_cell_29/recurrent_kernel* 
_output_shapes
:
´Ð*
dtype0

lstm_29/lstm_cell_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð**
shared_namelstm_29/lstm_cell_29/bias

-lstm_29/lstm_cell_29/bias/Read/ReadVariableOpReadVariableOplstm_29/lstm_cell_29/bias*
_output_shapes	
:Ð*
dtype0

lstm_30/lstm_cell_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*,
shared_namelstm_30/lstm_cell_30/kernel

/lstm_30/lstm_cell_30/kernel/Read/ReadVariableOpReadVariableOplstm_30/lstm_cell_30/kernel* 
_output_shapes
:
´Ð*
dtype0
¨
%lstm_30/lstm_cell_30/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*6
shared_name'%lstm_30/lstm_cell_30/recurrent_kernel
¡
9lstm_30/lstm_cell_30/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_30/lstm_cell_30/recurrent_kernel* 
_output_shapes
:
´Ð*
dtype0

lstm_30/lstm_cell_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð**
shared_namelstm_30/lstm_cell_30/bias

-lstm_30/lstm_cell_30/bias/Read/ReadVariableOpReadVariableOplstm_30/lstm_cell_30/bias*
_output_shapes	
:Ð*
dtype0

lstm_31/lstm_cell_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*,
shared_namelstm_31/lstm_cell_31/kernel

/lstm_31/lstm_cell_31/kernel/Read/ReadVariableOpReadVariableOplstm_31/lstm_cell_31/kernel* 
_output_shapes
:
´Ð*
dtype0
¨
%lstm_31/lstm_cell_31/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*6
shared_name'%lstm_31/lstm_cell_31/recurrent_kernel
¡
9lstm_31/lstm_cell_31/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_31/lstm_cell_31/recurrent_kernel* 
_output_shapes
:
´Ð*
dtype0

lstm_31/lstm_cell_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð**
shared_namelstm_31/lstm_cell_31/bias

-lstm_31/lstm_cell_31/bias/Read/ReadVariableOpReadVariableOplstm_31/lstm_cell_31/bias*
_output_shapes	
:Ð*
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

Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´´*'
shared_nameAdam/dense_31/kernel/m

*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m* 
_output_shapes
:
´´*
dtype0

Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:´*%
shared_nameAdam/dense_31/bias/m
z
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes	
:´*
dtype0

Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	´*'
shared_nameAdam/dense_32/kernel/m

*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m*
_output_shapes
:	´*
dtype0

Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_32/bias/m
y
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes
:*
dtype0
¡
"Adam/lstm_29/lstm_cell_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ð*3
shared_name$"Adam/lstm_29/lstm_cell_29/kernel/m

6Adam/lstm_29/lstm_cell_29/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_29/lstm_cell_29/kernel/m*
_output_shapes
:	Ð*
dtype0
¶
,Adam/lstm_29/lstm_cell_29/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*=
shared_name.,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m
¯
@Adam/lstm_29/lstm_cell_29/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m* 
_output_shapes
:
´Ð*
dtype0

 Adam/lstm_29/lstm_cell_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð*1
shared_name" Adam/lstm_29/lstm_cell_29/bias/m

4Adam/lstm_29/lstm_cell_29/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_29/lstm_cell_29/bias/m*
_output_shapes	
:Ð*
dtype0
¢
"Adam/lstm_30/lstm_cell_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*3
shared_name$"Adam/lstm_30/lstm_cell_30/kernel/m

6Adam/lstm_30/lstm_cell_30/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_30/lstm_cell_30/kernel/m* 
_output_shapes
:
´Ð*
dtype0
¶
,Adam/lstm_30/lstm_cell_30/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*=
shared_name.,Adam/lstm_30/lstm_cell_30/recurrent_kernel/m
¯
@Adam/lstm_30/lstm_cell_30/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_30/lstm_cell_30/recurrent_kernel/m* 
_output_shapes
:
´Ð*
dtype0

 Adam/lstm_30/lstm_cell_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð*1
shared_name" Adam/lstm_30/lstm_cell_30/bias/m

4Adam/lstm_30/lstm_cell_30/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_30/lstm_cell_30/bias/m*
_output_shapes	
:Ð*
dtype0
¢
"Adam/lstm_31/lstm_cell_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*3
shared_name$"Adam/lstm_31/lstm_cell_31/kernel/m

6Adam/lstm_31/lstm_cell_31/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_31/lstm_cell_31/kernel/m* 
_output_shapes
:
´Ð*
dtype0
¶
,Adam/lstm_31/lstm_cell_31/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*=
shared_name.,Adam/lstm_31/lstm_cell_31/recurrent_kernel/m
¯
@Adam/lstm_31/lstm_cell_31/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_31/lstm_cell_31/recurrent_kernel/m* 
_output_shapes
:
´Ð*
dtype0

 Adam/lstm_31/lstm_cell_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð*1
shared_name" Adam/lstm_31/lstm_cell_31/bias/m

4Adam/lstm_31/lstm_cell_31/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_31/lstm_cell_31/bias/m*
_output_shapes	
:Ð*
dtype0

Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´´*'
shared_nameAdam/dense_31/kernel/v

*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v* 
_output_shapes
:
´´*
dtype0

Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:´*%
shared_nameAdam/dense_31/bias/v
z
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes	
:´*
dtype0

Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	´*'
shared_nameAdam/dense_32/kernel/v

*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v*
_output_shapes
:	´*
dtype0

Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_32/bias/v
y
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes
:*
dtype0
¡
"Adam/lstm_29/lstm_cell_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ð*3
shared_name$"Adam/lstm_29/lstm_cell_29/kernel/v

6Adam/lstm_29/lstm_cell_29/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_29/lstm_cell_29/kernel/v*
_output_shapes
:	Ð*
dtype0
¶
,Adam/lstm_29/lstm_cell_29/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*=
shared_name.,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v
¯
@Adam/lstm_29/lstm_cell_29/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v* 
_output_shapes
:
´Ð*
dtype0

 Adam/lstm_29/lstm_cell_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð*1
shared_name" Adam/lstm_29/lstm_cell_29/bias/v

4Adam/lstm_29/lstm_cell_29/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_29/lstm_cell_29/bias/v*
_output_shapes	
:Ð*
dtype0
¢
"Adam/lstm_30/lstm_cell_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*3
shared_name$"Adam/lstm_30/lstm_cell_30/kernel/v

6Adam/lstm_30/lstm_cell_30/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_30/lstm_cell_30/kernel/v* 
_output_shapes
:
´Ð*
dtype0
¶
,Adam/lstm_30/lstm_cell_30/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*=
shared_name.,Adam/lstm_30/lstm_cell_30/recurrent_kernel/v
¯
@Adam/lstm_30/lstm_cell_30/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_30/lstm_cell_30/recurrent_kernel/v* 
_output_shapes
:
´Ð*
dtype0

 Adam/lstm_30/lstm_cell_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð*1
shared_name" Adam/lstm_30/lstm_cell_30/bias/v

4Adam/lstm_30/lstm_cell_30/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_30/lstm_cell_30/bias/v*
_output_shapes	
:Ð*
dtype0
¢
"Adam/lstm_31/lstm_cell_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*3
shared_name$"Adam/lstm_31/lstm_cell_31/kernel/v

6Adam/lstm_31/lstm_cell_31/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_31/lstm_cell_31/kernel/v* 
_output_shapes
:
´Ð*
dtype0
¶
,Adam/lstm_31/lstm_cell_31/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
´Ð*=
shared_name.,Adam/lstm_31/lstm_cell_31/recurrent_kernel/v
¯
@Adam/lstm_31/lstm_cell_31/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_31/lstm_cell_31/recurrent_kernel/v* 
_output_shapes
:
´Ð*
dtype0

 Adam/lstm_31/lstm_cell_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð*1
shared_name" Adam/lstm_31/lstm_cell_31/bias/v

4Adam/lstm_31/lstm_cell_31/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_31/lstm_cell_31/bias/v*
_output_shapes	
:Ð*
dtype0

NoOpNoOp
ÅS
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueöRBóR BìR
è
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
l
cell

state_spec
regularization_losses
	variables
trainable_variables
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
regularization_losses
	variables
trainable_variables
	keras_api
R
 trainable_variables
!	variables
"regularization_losses
#	keras_api
l
$cell
%
state_spec
&regularization_losses
'	variables
(trainable_variables
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
Ä
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate.mª/m«8m¬9m­Cm®Dm¯Em°Fm±Gm²Hm³Im´JmµKm¶.v·/v¸8v¹9vºCv»Dv¼Ev½Fv¾Gv¿HvÀIvÁJvÂKvÃ
 
^
C0
D1
E2
F3
G4
H5
I6
J7
K8
.9
/10
811
912
^
C0
D1
E2
F3
G4
H5
I6
J7
K8
.9
/10
811
912
­
regularization_losses
Llayer_regularization_losses
Mnon_trainable_variables
	variables
Nlayer_metrics
trainable_variables
Ometrics

Players
 

Q
state_size

Ckernel
Drecurrent_kernel
Ebias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
 
 

C0
D1
E2

C0
D1
E2
¹
regularization_losses
Vlayer_regularization_losses

Wstates
Xnon_trainable_variables
	variables
Ylayer_metrics
trainable_variables
Zmetrics

[layers
 
 
 
­
\layer_regularization_losses
]non_trainable_variables
trainable_variables
	variables
^layer_metrics
regularization_losses
_metrics

`layers

a
state_size

Fkernel
Grecurrent_kernel
Hbias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
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
¹
regularization_losses
flayer_regularization_losses

gstates
hnon_trainable_variables
	variables
ilayer_metrics
trainable_variables
jmetrics

klayers
 
 
 
­
llayer_regularization_losses
mnon_trainable_variables
 trainable_variables
!	variables
nlayer_metrics
"regularization_losses
ometrics

players

q
state_size

Ikernel
Jrecurrent_kernel
Kbias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
 
 

I0
J1
K2

I0
J1
K2
¹
&regularization_losses
vlayer_regularization_losses

wstates
xnon_trainable_variables
'	variables
ylayer_metrics
(trainable_variables
zmetrics

{layers
 
 
 
®
|layer_regularization_losses
}non_trainable_variables
*trainable_variables
+	variables
~layer_metrics
,regularization_losses
metrics
layers
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
²
 layer_regularization_losses
non_trainable_variables
0trainable_variables
1	variables
layer_metrics
2regularization_losses
metrics
layers
 
 
 
²
 layer_regularization_losses
non_trainable_variables
4trainable_variables
5	variables
layer_metrics
6regularization_losses
metrics
layers
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
²
 layer_regularization_losses
non_trainable_variables
:trainable_variables
;	variables
layer_metrics
<regularization_losses
metrics
layers
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
WU
VARIABLE_VALUElstm_29/lstm_cell_29/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_29/lstm_cell_29/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_29/lstm_cell_29/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_30/lstm_cell_30/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_30/lstm_cell_30/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_30/lstm_cell_30/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_31/lstm_cell_31/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_31/lstm_cell_31/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_31/lstm_cell_31/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
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

C0
D1
E2

C0
D1
E2
 
²
 layer_regularization_losses
non_trainable_variables
Rtrainable_variables
S	variables
layer_metrics
Tregularization_losses
metrics
layers
 
 
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

F0
G1
H2

F0
G1
H2
 
²
 layer_regularization_losses
non_trainable_variables
btrainable_variables
c	variables
layer_metrics
dregularization_losses
metrics
layers
 
 
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

I0
J1
K2

I0
J1
K2
 
²
 layer_regularization_losses
non_trainable_variables
rtrainable_variables
s	variables
layer_metrics
tregularization_losses
metrics
 layers
 
 
 
 
 

$0
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

¡total

¢count
£	variables
¤	keras_api
I

¥total

¦count
§
_fn_kwargs
¨	variables
©	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

¡0
¢1

£	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

¥0
¦1

¨	variables
~|
VARIABLE_VALUEAdam/dense_31/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_29/lstm_cell_29/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_29/lstm_cell_29/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_29/lstm_cell_29/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_30/lstm_cell_30/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_30/lstm_cell_30/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_30/lstm_cell_30/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_31/lstm_cell_31/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_31/lstm_cell_31/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_31/lstm_cell_31/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_29/lstm_cell_29/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_29/lstm_cell_29/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_29/lstm_cell_29/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_30/lstm_cell_30/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_30/lstm_cell_30/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_30/lstm_cell_30/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_31/lstm_cell_31/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_31/lstm_cell_31/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_31/lstm_cell_31/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_29_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
´
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_29_inputlstm_29/lstm_cell_29/kernel%lstm_29/lstm_cell_29/recurrent_kernellstm_29/lstm_cell_29/biaslstm_30/lstm_cell_30/kernel%lstm_30/lstm_cell_30/recurrent_kernellstm_30/lstm_cell_30/biaslstm_31/lstm_cell_31/kernel%lstm_31/lstm_cell_31/recurrent_kernellstm_31/lstm_cell_31/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1678154
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¾
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_29/lstm_cell_29/kernel/Read/ReadVariableOp9lstm_29/lstm_cell_29/recurrent_kernel/Read/ReadVariableOp-lstm_29/lstm_cell_29/bias/Read/ReadVariableOp/lstm_30/lstm_cell_30/kernel/Read/ReadVariableOp9lstm_30/lstm_cell_30/recurrent_kernel/Read/ReadVariableOp-lstm_30/lstm_cell_30/bias/Read/ReadVariableOp/lstm_31/lstm_cell_31/kernel/Read/ReadVariableOp9lstm_31/lstm_cell_31/recurrent_kernel/Read/ReadVariableOp-lstm_31/lstm_cell_31/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp6Adam/lstm_29/lstm_cell_29/kernel/m/Read/ReadVariableOp@Adam/lstm_29/lstm_cell_29/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_29/lstm_cell_29/bias/m/Read/ReadVariableOp6Adam/lstm_30/lstm_cell_30/kernel/m/Read/ReadVariableOp@Adam/lstm_30/lstm_cell_30/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_30/lstm_cell_30/bias/m/Read/ReadVariableOp6Adam/lstm_31/lstm_cell_31/kernel/m/Read/ReadVariableOp@Adam/lstm_31/lstm_cell_31/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_31/lstm_cell_31/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp6Adam/lstm_29/lstm_cell_29/kernel/v/Read/ReadVariableOp@Adam/lstm_29/lstm_cell_29/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_29/lstm_cell_29/bias/v/Read/ReadVariableOp6Adam/lstm_30/lstm_cell_30/kernel/v/Read/ReadVariableOp@Adam/lstm_30/lstm_cell_30/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_30/lstm_cell_30/bias/v/Read/ReadVariableOp6Adam/lstm_31/lstm_cell_31/kernel/v/Read/ReadVariableOp@Adam/lstm_31/lstm_cell_31/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_31/lstm_cell_31/bias/v/Read/ReadVariableOpConst*=
Tin6
422	*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1681696
ù
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_31/kerneldense_31/biasdense_32/kerneldense_32/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_29/lstm_cell_29/kernel%lstm_29/lstm_cell_29/recurrent_kernellstm_29/lstm_cell_29/biaslstm_30/lstm_cell_30/kernel%lstm_30/lstm_cell_30/recurrent_kernellstm_30/lstm_cell_30/biaslstm_31/lstm_cell_31/kernel%lstm_31/lstm_cell_31/recurrent_kernellstm_31/lstm_cell_31/biastotalcounttotal_1count_1Adam/dense_31/kernel/mAdam/dense_31/bias/mAdam/dense_32/kernel/mAdam/dense_32/bias/m"Adam/lstm_29/lstm_cell_29/kernel/m,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m Adam/lstm_29/lstm_cell_29/bias/m"Adam/lstm_30/lstm_cell_30/kernel/m,Adam/lstm_30/lstm_cell_30/recurrent_kernel/m Adam/lstm_30/lstm_cell_30/bias/m"Adam/lstm_31/lstm_cell_31/kernel/m,Adam/lstm_31/lstm_cell_31/recurrent_kernel/m Adam/lstm_31/lstm_cell_31/bias/mAdam/dense_31/kernel/vAdam/dense_31/bias/vAdam/dense_32/kernel/vAdam/dense_32/bias/v"Adam/lstm_29/lstm_cell_29/kernel/v,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v Adam/lstm_29/lstm_cell_29/bias/v"Adam/lstm_30/lstm_cell_30/kernel/v,Adam/lstm_30/lstm_cell_30/recurrent_kernel/v Adam/lstm_30/lstm_cell_30/bias/v"Adam/lstm_31/lstm_cell_31/kernel/v,Adam/lstm_31/lstm_cell_31/recurrent_kernel/v Adam/lstm_31/lstm_cell_31/bias/v*<
Tin5
321*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1681850­È6
V
 
D__inference_lstm_31_layer_call_and_return_conditional_losses_1680816
inputs_0?
+lstm_cell_31_matmul_readvariableop_resource:
´ÐA
-lstm_cell_31_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_31_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_31/BiasAdd/ReadVariableOp¢"lstm_cell_31/MatMul/ReadVariableOp¢$lstm_cell_31/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp­
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul¼
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp©
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul_1 
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/add´
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp­
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim÷
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_31/split
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_1
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_1
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/add_1
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu_1¡
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1680732*
condR
while_cond_1680731*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
inputs/0
Íf
ï
 __inference__traced_save_1681696
file_prefix.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_29_lstm_cell_29_kernel_read_readvariableopD
@savev2_lstm_29_lstm_cell_29_recurrent_kernel_read_readvariableop8
4savev2_lstm_29_lstm_cell_29_bias_read_readvariableop:
6savev2_lstm_30_lstm_cell_30_kernel_read_readvariableopD
@savev2_lstm_30_lstm_cell_30_recurrent_kernel_read_readvariableop8
4savev2_lstm_30_lstm_cell_30_bias_read_readvariableop:
6savev2_lstm_31_lstm_cell_31_kernel_read_readvariableopD
@savev2_lstm_31_lstm_cell_31_recurrent_kernel_read_readvariableop8
4savev2_lstm_31_lstm_cell_31_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableopA
=savev2_adam_lstm_29_lstm_cell_29_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_29_lstm_cell_29_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_29_lstm_cell_29_bias_m_read_readvariableopA
=savev2_adam_lstm_30_lstm_cell_30_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_30_lstm_cell_30_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_30_lstm_cell_30_bias_m_read_readvariableopA
=savev2_adam_lstm_31_lstm_cell_31_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_31_lstm_cell_31_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_31_lstm_cell_31_bias_m_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableopA
=savev2_adam_lstm_29_lstm_cell_29_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_29_lstm_cell_29_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_29_lstm_cell_29_bias_v_read_readvariableopA
=savev2_adam_lstm_30_lstm_cell_30_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_30_lstm_cell_30_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_30_lstm_cell_30_bias_v_read_readvariableopA
=savev2_adam_lstm_31_lstm_cell_31_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_31_lstm_cell_31_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_31_lstm_cell_31_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*
valueB1B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesê
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¬
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_29_lstm_cell_29_kernel_read_readvariableop@savev2_lstm_29_lstm_cell_29_recurrent_kernel_read_readvariableop4savev2_lstm_29_lstm_cell_29_bias_read_readvariableop6savev2_lstm_30_lstm_cell_30_kernel_read_readvariableop@savev2_lstm_30_lstm_cell_30_recurrent_kernel_read_readvariableop4savev2_lstm_30_lstm_cell_30_bias_read_readvariableop6savev2_lstm_31_lstm_cell_31_kernel_read_readvariableop@savev2_lstm_31_lstm_cell_31_recurrent_kernel_read_readvariableop4savev2_lstm_31_lstm_cell_31_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableop=savev2_adam_lstm_29_lstm_cell_29_kernel_m_read_readvariableopGsavev2_adam_lstm_29_lstm_cell_29_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_29_lstm_cell_29_bias_m_read_readvariableop=savev2_adam_lstm_30_lstm_cell_30_kernel_m_read_readvariableopGsavev2_adam_lstm_30_lstm_cell_30_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_30_lstm_cell_30_bias_m_read_readvariableop=savev2_adam_lstm_31_lstm_cell_31_kernel_m_read_readvariableopGsavev2_adam_lstm_31_lstm_cell_31_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_31_lstm_cell_31_bias_m_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableop=savev2_adam_lstm_29_lstm_cell_29_kernel_v_read_readvariableopGsavev2_adam_lstm_29_lstm_cell_29_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_29_lstm_cell_29_bias_v_read_readvariableop=savev2_adam_lstm_30_lstm_cell_30_kernel_v_read_readvariableopGsavev2_adam_lstm_30_lstm_cell_30_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_30_lstm_cell_30_bias_v_read_readvariableop=savev2_adam_lstm_31_lstm_cell_31_kernel_v_read_readvariableopGsavev2_adam_lstm_31_lstm_cell_31_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_31_lstm_cell_31_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes5
321	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*«
_input_shapes
: :
´´:´:	´:: : : : : :	Ð:
´Ð:Ð:
´Ð:
´Ð:Ð:
´Ð:
´Ð:Ð: : : : :
´´:´:	´::	Ð:
´Ð:Ð:
´Ð:
´Ð:Ð:
´Ð:
´Ð:Ð:
´´:´:	´::	Ð:
´Ð:Ð:
´Ð:
´Ð:Ð:
´Ð:
´Ð:Ð: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
´´:!

_output_shapes	
:´:%!

_output_shapes
:	´: 
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
:	Ð:&"
 
_output_shapes
:
´Ð:!

_output_shapes	
:Ð:&"
 
_output_shapes
:
´Ð:&"
 
_output_shapes
:
´Ð:!

_output_shapes	
:Ð:&"
 
_output_shapes
:
´Ð:&"
 
_output_shapes
:
´Ð:!

_output_shapes	
:Ð:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
´´:!

_output_shapes	
:´:%!

_output_shapes
:	´: 

_output_shapes
::%!

_output_shapes
:	Ð:&"
 
_output_shapes
:
´Ð:!

_output_shapes	
:Ð:&"
 
_output_shapes
:
´Ð:&"
 
_output_shapes
:
´Ð:! 

_output_shapes	
:Ð:&!"
 
_output_shapes
:
´Ð:&""
 
_output_shapes
:
´Ð:!#

_output_shapes	
:Ð:&$"
 
_output_shapes
:
´´:!%

_output_shapes	
:´:%&!

_output_shapes
:	´: '

_output_shapes
::%(!

_output_shapes
:	Ð:&)"
 
_output_shapes
:
´Ð:!*

_output_shapes	
:Ð:&+"
 
_output_shapes
:
´Ð:&,"
 
_output_shapes
:
´Ð:!-

_output_shapes	
:Ð:&."
 
_output_shapes
:
´Ð:&/"
 
_output_shapes
:
´Ð:!0

_output_shapes	
:Ð:1

_output_shapes
: 
½U

D__inference_lstm_29_layer_call_and_return_conditional_losses_1679816

inputs>
+lstm_cell_29_matmul_readvariableop_resource:	ÐA
-lstm_cell_29_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_29_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_29/BiasAdd/ReadVariableOp¢"lstm_cell_29/MatMul/ReadVariableOp¢$lstm_cell_29/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp­
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul¼
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp©
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul_1 
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/add´
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp­
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim÷
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_29/split
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_1
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_1
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/add_1
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu_1¡
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1679732*
condR
while_cond_1679731*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
f
G__inference_dropout_49_layer_call_and_return_conditional_losses_1677361

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
äJ
Ó

lstm_29_while_body_1678275,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3+
'lstm_29_while_lstm_29_strided_slice_1_0g
clstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0:	ÐQ
=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0:
´ÐK
<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0:	Ð
lstm_29_while_identity
lstm_29_while_identity_1
lstm_29_while_identity_2
lstm_29_while_identity_3
lstm_29_while_identity_4
lstm_29_while_identity_5)
%lstm_29_while_lstm_29_strided_slice_1e
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorL
9lstm_29_while_lstm_cell_29_matmul_readvariableop_resource:	ÐO
;lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐI
:lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource:	Ð¢1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp¢0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp¢2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpÓ
?lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_29/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0lstm_29_while_placeholderHlstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_29/while/TensorArrayV2Read/TensorListGetItemá
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	Ð*
dtype022
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp÷
!lstm_29/while/lstm_cell_29/MatMulMatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2#
!lstm_29/while/lstm_cell_29/MatMulè
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype024
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpà
#lstm_29/while/lstm_cell_29/MatMul_1MatMullstm_29_while_placeholder_2:lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2%
#lstm_29/while/lstm_cell_29/MatMul_1Ø
lstm_29/while/lstm_cell_29/addAddV2+lstm_29/while/lstm_cell_29/MatMul:product:0-lstm_29/while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2 
lstm_29/while/lstm_cell_29/addà
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype023
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOpå
"lstm_29/while/lstm_cell_29/BiasAddBiasAdd"lstm_29/while/lstm_cell_29/add:z:09lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2$
"lstm_29/while/lstm_cell_29/BiasAdd
*lstm_29/while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_29/while/lstm_cell_29/split/split_dim¯
 lstm_29/while/lstm_cell_29/splitSplit3lstm_29/while/lstm_cell_29/split/split_dim:output:0+lstm_29/while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2"
 lstm_29/while/lstm_cell_29/split±
"lstm_29/while/lstm_cell_29/SigmoidSigmoid)lstm_29/while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2$
"lstm_29/while/lstm_cell_29/Sigmoidµ
$lstm_29/while/lstm_cell_29/Sigmoid_1Sigmoid)lstm_29/while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_29/while/lstm_cell_29/Sigmoid_1Á
lstm_29/while/lstm_cell_29/mulMul(lstm_29/while/lstm_cell_29/Sigmoid_1:y:0lstm_29_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_29/while/lstm_cell_29/mul¨
lstm_29/while/lstm_cell_29/ReluRelu)lstm_29/while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
lstm_29/while/lstm_cell_29/ReluÕ
 lstm_29/while/lstm_cell_29/mul_1Mul&lstm_29/while/lstm_cell_29/Sigmoid:y:0-lstm_29/while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_29/while/lstm_cell_29/mul_1Ê
 lstm_29/while/lstm_cell_29/add_1AddV2"lstm_29/while/lstm_cell_29/mul:z:0$lstm_29/while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_29/while/lstm_cell_29/add_1µ
$lstm_29/while/lstm_cell_29/Sigmoid_2Sigmoid)lstm_29/while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_29/while/lstm_cell_29/Sigmoid_2§
!lstm_29/while/lstm_cell_29/Relu_1Relu$lstm_29/while/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!lstm_29/while/lstm_cell_29/Relu_1Ù
 lstm_29/while/lstm_cell_29/mul_2Mul(lstm_29/while/lstm_cell_29/Sigmoid_2:y:0/lstm_29/while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_29/while/lstm_cell_29/mul_2
2lstm_29/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_29_while_placeholder_1lstm_29_while_placeholder$lstm_29/while/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_29/while/TensorArrayV2Write/TensorListSetIteml
lstm_29/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_29/while/add/y
lstm_29/while/addAddV2lstm_29_while_placeholderlstm_29/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_29/while/addp
lstm_29/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_29/while/add_1/y
lstm_29/while/add_1AddV2(lstm_29_while_lstm_29_while_loop_counterlstm_29/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_29/while/add_1
lstm_29/while/IdentityIdentitylstm_29/while/add_1:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity¦
lstm_29/while/Identity_1Identity.lstm_29_while_lstm_29_while_maximum_iterations^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_1
lstm_29/while/Identity_2Identitylstm_29/while/add:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_2º
lstm_29/while/Identity_3IdentityBlstm_29/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_3®
lstm_29/while/Identity_4Identity$lstm_29/while/lstm_cell_29/mul_2:z:0^lstm_29/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/while/Identity_4®
lstm_29/while/Identity_5Identity$lstm_29/while/lstm_cell_29/add_1:z:0^lstm_29/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/while/Identity_5
lstm_29/while/NoOpNoOp2^lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp1^lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp3^lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_29/while/NoOp"9
lstm_29_while_identitylstm_29/while/Identity:output:0"=
lstm_29_while_identity_1!lstm_29/while/Identity_1:output:0"=
lstm_29_while_identity_2!lstm_29/while/Identity_2:output:0"=
lstm_29_while_identity_3!lstm_29/while/Identity_3:output:0"=
lstm_29_while_identity_4!lstm_29/while/Identity_4:output:0"=
lstm_29_while_identity_5!lstm_29/while/Identity_5:output:0"P
%lstm_29_while_lstm_29_strided_slice_1'lstm_29_while_lstm_29_strided_slice_1_0"z
:lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0"|
;lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0"x
9lstm_29_while_lstm_cell_29_matmul_readvariableop_resource;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0"È
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2f
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp2d
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp2h
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
ô

(sequential_13_lstm_31_while_cond_1674776H
Dsequential_13_lstm_31_while_sequential_13_lstm_31_while_loop_counterN
Jsequential_13_lstm_31_while_sequential_13_lstm_31_while_maximum_iterations+
'sequential_13_lstm_31_while_placeholder-
)sequential_13_lstm_31_while_placeholder_1-
)sequential_13_lstm_31_while_placeholder_2-
)sequential_13_lstm_31_while_placeholder_3J
Fsequential_13_lstm_31_while_less_sequential_13_lstm_31_strided_slice_1a
]sequential_13_lstm_31_while_sequential_13_lstm_31_while_cond_1674776___redundant_placeholder0a
]sequential_13_lstm_31_while_sequential_13_lstm_31_while_cond_1674776___redundant_placeholder1a
]sequential_13_lstm_31_while_sequential_13_lstm_31_while_cond_1674776___redundant_placeholder2a
]sequential_13_lstm_31_while_sequential_13_lstm_31_while_cond_1674776___redundant_placeholder3(
$sequential_13_lstm_31_while_identity
Þ
 sequential_13/lstm_31/while/LessLess'sequential_13_lstm_31_while_placeholderFsequential_13_lstm_31_while_less_sequential_13_lstm_31_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_13/lstm_31/while/Less
$sequential_13/lstm_31/while/IdentityIdentity$sequential_13/lstm_31/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_13/lstm_31/while/Identity"U
$sequential_13_lstm_31_while_identity-sequential_13/lstm_31/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
ÿ?

D__inference_lstm_31_layer_call_and_return_conditional_losses_1676464

inputs(
lstm_cell_31_1676382:
´Ð(
lstm_cell_31_1676384:
´Ð#
lstm_cell_31_1676386:	Ð
identity¢$lstm_cell_31/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¤
$lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_31_1676382lstm_cell_31_1676384lstm_cell_31_1676386*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_16763252&
$lstm_cell_31/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÉ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_31_1676382lstm_cell_31_1676384lstm_cell_31_1676386*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1676395*
condR
while_cond_1676394*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identity}
NoOpNoOp%^lstm_cell_31/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 2L
$lstm_cell_31/StatefulPartitionedCall$lstm_cell_31/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
äJ
Ó

lstm_29_while_body_1678753,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3+
'lstm_29_while_lstm_29_strided_slice_1_0g
clstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0:	ÐQ
=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0:
´ÐK
<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0:	Ð
lstm_29_while_identity
lstm_29_while_identity_1
lstm_29_while_identity_2
lstm_29_while_identity_3
lstm_29_while_identity_4
lstm_29_while_identity_5)
%lstm_29_while_lstm_29_strided_slice_1e
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorL
9lstm_29_while_lstm_cell_29_matmul_readvariableop_resource:	ÐO
;lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐI
:lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource:	Ð¢1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp¢0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp¢2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpÓ
?lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_29/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0lstm_29_while_placeholderHlstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_29/while/TensorArrayV2Read/TensorListGetItemá
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	Ð*
dtype022
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp÷
!lstm_29/while/lstm_cell_29/MatMulMatMul8lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2#
!lstm_29/while/lstm_cell_29/MatMulè
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype024
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpà
#lstm_29/while/lstm_cell_29/MatMul_1MatMullstm_29_while_placeholder_2:lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2%
#lstm_29/while/lstm_cell_29/MatMul_1Ø
lstm_29/while/lstm_cell_29/addAddV2+lstm_29/while/lstm_cell_29/MatMul:product:0-lstm_29/while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2 
lstm_29/while/lstm_cell_29/addà
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype023
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOpå
"lstm_29/while/lstm_cell_29/BiasAddBiasAdd"lstm_29/while/lstm_cell_29/add:z:09lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2$
"lstm_29/while/lstm_cell_29/BiasAdd
*lstm_29/while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_29/while/lstm_cell_29/split/split_dim¯
 lstm_29/while/lstm_cell_29/splitSplit3lstm_29/while/lstm_cell_29/split/split_dim:output:0+lstm_29/while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2"
 lstm_29/while/lstm_cell_29/split±
"lstm_29/while/lstm_cell_29/SigmoidSigmoid)lstm_29/while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2$
"lstm_29/while/lstm_cell_29/Sigmoidµ
$lstm_29/while/lstm_cell_29/Sigmoid_1Sigmoid)lstm_29/while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_29/while/lstm_cell_29/Sigmoid_1Á
lstm_29/while/lstm_cell_29/mulMul(lstm_29/while/lstm_cell_29/Sigmoid_1:y:0lstm_29_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_29/while/lstm_cell_29/mul¨
lstm_29/while/lstm_cell_29/ReluRelu)lstm_29/while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
lstm_29/while/lstm_cell_29/ReluÕ
 lstm_29/while/lstm_cell_29/mul_1Mul&lstm_29/while/lstm_cell_29/Sigmoid:y:0-lstm_29/while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_29/while/lstm_cell_29/mul_1Ê
 lstm_29/while/lstm_cell_29/add_1AddV2"lstm_29/while/lstm_cell_29/mul:z:0$lstm_29/while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_29/while/lstm_cell_29/add_1µ
$lstm_29/while/lstm_cell_29/Sigmoid_2Sigmoid)lstm_29/while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_29/while/lstm_cell_29/Sigmoid_2§
!lstm_29/while/lstm_cell_29/Relu_1Relu$lstm_29/while/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!lstm_29/while/lstm_cell_29/Relu_1Ù
 lstm_29/while/lstm_cell_29/mul_2Mul(lstm_29/while/lstm_cell_29/Sigmoid_2:y:0/lstm_29/while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_29/while/lstm_cell_29/mul_2
2lstm_29/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_29_while_placeholder_1lstm_29_while_placeholder$lstm_29/while/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_29/while/TensorArrayV2Write/TensorListSetIteml
lstm_29/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_29/while/add/y
lstm_29/while/addAddV2lstm_29_while_placeholderlstm_29/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_29/while/addp
lstm_29/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_29/while/add_1/y
lstm_29/while/add_1AddV2(lstm_29_while_lstm_29_while_loop_counterlstm_29/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_29/while/add_1
lstm_29/while/IdentityIdentitylstm_29/while/add_1:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity¦
lstm_29/while/Identity_1Identity.lstm_29_while_lstm_29_while_maximum_iterations^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_1
lstm_29/while/Identity_2Identitylstm_29/while/add:z:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_2º
lstm_29/while/Identity_3IdentityBlstm_29/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_29/while/NoOp*
T0*
_output_shapes
: 2
lstm_29/while/Identity_3®
lstm_29/while/Identity_4Identity$lstm_29/while/lstm_cell_29/mul_2:z:0^lstm_29/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/while/Identity_4®
lstm_29/while/Identity_5Identity$lstm_29/while/lstm_cell_29/add_1:z:0^lstm_29/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/while/Identity_5
lstm_29/while/NoOpNoOp2^lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp1^lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp3^lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_29/while/NoOp"9
lstm_29_while_identitylstm_29/while/Identity:output:0"=
lstm_29_while_identity_1!lstm_29/while/Identity_1:output:0"=
lstm_29_while_identity_2!lstm_29/while/Identity_2:output:0"=
lstm_29_while_identity_3!lstm_29/while/Identity_3:output:0"=
lstm_29_while_identity_4!lstm_29/while/Identity_4:output:0"=
lstm_29_while_identity_5!lstm_29/while/Identity_5:output:0"P
%lstm_29_while_lstm_29_strided_slice_1'lstm_29_while_lstm_29_strided_slice_1_0"z
:lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource<lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0"|
;lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource=lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0"x
9lstm_29_while_lstm_cell_29_matmul_readvariableop_resource;lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0"È
alstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensorclstm_29_while_tensorarrayv2read_tensorlistgetitem_lstm_29_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2f
1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp1lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp2d
0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp0lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp2h
2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp2lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
·
¸
)__inference_lstm_31_layer_call_fn_1680519

inputs
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_16771732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs


I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1681399

inputs
states_0
states_12
matmul_readvariableop_resource:
´Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
Þ
È
while_cond_1679945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1679945___redundant_placeholder05
1while_while_cond_1679945___redundant_placeholder15
1while_while_cond_1679945___redundant_placeholder25
1while_while_cond_1679945___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
Þ
È
while_cond_1676192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1676192___redundant_placeholder05
1while_while_cond_1676192___redundant_placeholder15
1while_while_cond_1676192___redundant_placeholder25
1while_while_cond_1676192___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
¹
e
,__inference_dropout_49_layer_call_fn_1681112

inputs
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_16773612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Ï

è
lstm_31_while_cond_1679046,
(lstm_31_while_lstm_31_while_loop_counter2
.lstm_31_while_lstm_31_while_maximum_iterations
lstm_31_while_placeholder
lstm_31_while_placeholder_1
lstm_31_while_placeholder_2
lstm_31_while_placeholder_3.
*lstm_31_while_less_lstm_31_strided_slice_1E
Alstm_31_while_lstm_31_while_cond_1679046___redundant_placeholder0E
Alstm_31_while_lstm_31_while_cond_1679046___redundant_placeholder1E
Alstm_31_while_lstm_31_while_cond_1679046___redundant_placeholder2E
Alstm_31_while_lstm_31_while_cond_1679046___redundant_placeholder3
lstm_31_while_identity

lstm_31/while/LessLesslstm_31_while_placeholder*lstm_31_while_less_lstm_31_strided_slice_1*
T0*
_output_shapes
: 2
lstm_31/while/Lessu
lstm_31/while/IdentityIdentitylstm_31/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_31/while/Identity"9
lstm_31_while_identitylstm_31/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
³?
Õ
while_body_1680732
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_31_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_31/BiasAdd/ReadVariableOp¢(while/lstm_cell_31/MatMul/ReadVariableOp¢*while/lstm_cell_31/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp×
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMulÐ
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOpÀ
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMul_1¸
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/addÈ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOpÅ
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/BiasAdd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_31/split
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_1¡
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Reluµ
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_1ª
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/add_1
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_2
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Relu_1¹
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
÷
ß
/__inference_sequential_13_layer_call_fn_1678185

inputs
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
	unknown_2:
´Ð
	unknown_3:
´Ð
	unknown_4:	Ð
	unknown_5:
´Ð
	unknown_6:
´Ð
	unknown_7:	Ð
	unknown_8:
´´
	unknown_9:	´

unknown_10:	´

unknown_11:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_16772692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÃU

D__inference_lstm_31_layer_call_and_return_conditional_losses_1677520

inputs?
+lstm_cell_31_matmul_readvariableop_resource:
´ÐA
-lstm_cell_31_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_31_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_31/BiasAdd/ReadVariableOp¢"lstm_cell_31/MatMul/ReadVariableOp¢$lstm_cell_31/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp­
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul¼
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp©
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul_1 
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/add´
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp­
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim÷
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_31/split
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_1
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_1
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/add_1
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu_1¡
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1677436*
condR
while_cond_1677435*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
ÃU

D__inference_lstm_31_layer_call_and_return_conditional_losses_1677173

inputs?
+lstm_cell_31_matmul_readvariableop_resource:
´ÐA
-lstm_cell_31_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_31_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_31/BiasAdd/ReadVariableOp¢"lstm_cell_31/MatMul/ReadVariableOp¢$lstm_cell_31/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp­
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul¼
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp©
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul_1 
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/add´
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp­
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim÷
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_31/split
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_1
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_1
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/add_1
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu_1¡
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1677089*
condR
while_cond_1677088*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs

e
G__inference_dropout_47_layer_call_and_return_conditional_losses_1679831

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
³?
Õ
while_body_1677436
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_31_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_31/BiasAdd/ReadVariableOp¢(while/lstm_cell_31/MatMul/ReadVariableOp¢*while/lstm_cell_31/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp×
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMulÐ
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOpÀ
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMul_1¸
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/addÈ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOpÅ
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/BiasAdd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_31/split
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_1¡
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Reluµ
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_1ª
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/add_1
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_2
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Relu_1¹
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
¹
e
,__inference_dropout_50_layer_call_fn_1681179

inputs
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_16773282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
÷%
î
while_body_1675797
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_30_1675821_0:
´Ð0
while_lstm_cell_30_1675823_0:
´Ð+
while_lstm_cell_30_1675825_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_30_1675821:
´Ð.
while_lstm_cell_30_1675823:
´Ð)
while_lstm_cell_30_1675825:	Ð¢*while/lstm_cell_30/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemè
*while/lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_30_1675821_0while_lstm_cell_30_1675823_0while_lstm_cell_30_1675825_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_16757272,
*while/lstm_cell_30/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_30/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_30/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_30/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_30_1675821while_lstm_cell_30_1675821_0":
while_lstm_cell_30_1675823while_lstm_cell_30_1675823_0":
while_lstm_cell_30_1675825while_lstm_cell_30_1675825_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2X
*while/lstm_cell_30/StatefulPartitionedCall*while/lstm_cell_30/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
¯?
Ó
while_body_1679589
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	ÐI
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	ÐG
3while_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_29_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_29/BiasAdd/ReadVariableOp¢(while/lstm_cell_29/MatMul/ReadVariableOp¢*while/lstm_cell_29/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	Ð*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp×
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMulÐ
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOpÀ
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMul_1¸
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/addÈ
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOpÅ
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/BiasAdd
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_29/split
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_1¡
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Reluµ
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_1ª
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/add_1
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_2
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Relu_1¹
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1680088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1680088___redundant_placeholder05
1while_while_cond_1680088___redundant_placeholder15
1while_while_cond_1680088___redundant_placeholder25
1while_while_cond_1680088___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
ô

(sequential_13_lstm_30_while_cond_1674636H
Dsequential_13_lstm_30_while_sequential_13_lstm_30_while_loop_counterN
Jsequential_13_lstm_30_while_sequential_13_lstm_30_while_maximum_iterations+
'sequential_13_lstm_30_while_placeholder-
)sequential_13_lstm_30_while_placeholder_1-
)sequential_13_lstm_30_while_placeholder_2-
)sequential_13_lstm_30_while_placeholder_3J
Fsequential_13_lstm_30_while_less_sequential_13_lstm_30_strided_slice_1a
]sequential_13_lstm_30_while_sequential_13_lstm_30_while_cond_1674636___redundant_placeholder0a
]sequential_13_lstm_30_while_sequential_13_lstm_30_while_cond_1674636___redundant_placeholder1a
]sequential_13_lstm_30_while_sequential_13_lstm_30_while_cond_1674636___redundant_placeholder2a
]sequential_13_lstm_30_while_sequential_13_lstm_30_while_cond_1674636___redundant_placeholder3(
$sequential_13_lstm_30_while_identity
Þ
 sequential_13/lstm_30/while/LessLess'sequential_13_lstm_30_while_placeholderFsequential_13_lstm_30_while_less_sequential_13_lstm_30_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_13/lstm_30/while/Less
$sequential_13/lstm_30/while/IdentityIdentity$sequential_13/lstm_30/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_13/lstm_30/while/Identity"U
$sequential_13_lstm_30_while_identity-sequential_13/lstm_30/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
³?
Õ
while_body_1676932
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_30_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_30/BiasAdd/ReadVariableOp¢(while/lstm_cell_30/MatMul/ReadVariableOp¢*while/lstm_cell_30/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp×
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMulÐ
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOpÀ
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMul_1¸
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/addÈ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOpÅ
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/BiasAdd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_30/split
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_1¡
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Reluµ
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_1ª
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/add_1
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_2
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Relu_1¹
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
½U

D__inference_lstm_29_layer_call_and_return_conditional_losses_1679673

inputs>
+lstm_cell_29_matmul_readvariableop_resource:	ÐA
-lstm_cell_29_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_29_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_29/BiasAdd/ReadVariableOp¢"lstm_cell_29/MatMul/ReadVariableOp¢$lstm_cell_29/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp­
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul¼
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp©
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul_1 
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/add´
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp­
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim÷
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_29/split
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_1
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_1
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/add_1
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu_1¡
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1679589*
condR
while_cond_1679588*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÃU

D__inference_lstm_30_layer_call_and_return_conditional_losses_1677016

inputs?
+lstm_cell_30_matmul_readvariableop_resource:
´ÐA
-lstm_cell_30_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_30_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_30/BiasAdd/ReadVariableOp¢"lstm_cell_30/MatMul/ReadVariableOp¢$lstm_cell_30/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp­
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul¼
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp©
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul_1 
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/add´
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp­
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim÷
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_30/split
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_1
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_1
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/add_1
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu_1¡
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1676932*
condR
while_cond_1676931*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Ö
f
G__inference_dropout_48_layer_call_and_return_conditional_losses_1680486

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Å
ø
.__inference_lstm_cell_29_layer_call_fn_1681269

inputs
states_0
states_1
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity

identity_1

identity_2¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_16751292
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
´
·
)__inference_lstm_29_layer_call_fn_1679244

inputs
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_16778962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯?
Ó
while_body_1679303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	ÐI
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	ÐG
3while_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_29_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_29/BiasAdd/ReadVariableOp¢(while/lstm_cell_29/MatMul/ReadVariableOp¢*while/lstm_cell_29/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	Ð*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp×
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMulÐ
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOpÀ
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMul_1¸
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/addÈ
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOpÅ
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/BiasAdd
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_29/split
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_1¡
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Reluµ
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_1ª
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/add_1
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_2
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Relu_1¹
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
·
¸
)__inference_lstm_31_layer_call_fn_1680530

inputs
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_16775202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Þ
È
while_cond_1680588
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1680588___redundant_placeholder05
1while_while_cond_1680588___redundant_placeholder15
1while_while_cond_1680588___redundant_placeholder25
1while_while_cond_1680588___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
³?
Õ
while_body_1680875
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_31_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_31/BiasAdd/ReadVariableOp¢(while/lstm_cell_31/MatMul/ReadVariableOp¢*while/lstm_cell_31/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp×
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMulÐ
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOpÀ
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMul_1¸
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/addÈ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOpÅ
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/BiasAdd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_31/split
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_1¡
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Reluµ
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_1ª
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/add_1
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_2
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Relu_1¹
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1680374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1680374___redundant_placeholder05
1while_while_cond_1680374___redundant_placeholder15
1while_while_cond_1680374___redundant_placeholder25
1while_while_cond_1680374___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
Ö!
ÿ
E__inference_dense_31_layer_call_and_return_conditional_losses_1677219

inputs5
!tensordot_readvariableop_resource:
´´.
biasadd_readvariableop_resource:	´
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
´´*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:´2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:´*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
ú?

D__inference_lstm_29_layer_call_and_return_conditional_losses_1675066

inputs'
lstm_cell_29_1674984:	Ð(
lstm_cell_29_1674986:
´Ð#
lstm_cell_29_1674988:	Ð
identity¢$lstm_cell_29/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¤
$lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_29_1674984lstm_cell_29_1674986lstm_cell_29_1674988*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_16749832&
$lstm_cell_29/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÉ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_29_1674984lstm_cell_29_1674986lstm_cell_29_1674988*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1674997*
condR
while_cond_1674996*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identity}
NoOpNoOp%^lstm_cell_29/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_29/StatefulPartitionedCall$lstm_cell_29/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
È
while_cond_1675796
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1675796___redundant_placeholder05
1while_while_cond_1675796___redundant_placeholder15
1while_while_cond_1675796___redundant_placeholder25
1while_while_cond_1675796___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
³?
Õ
while_body_1677624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_30_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_30/BiasAdd/ReadVariableOp¢(while/lstm_cell_30/MatMul/ReadVariableOp¢*while/lstm_cell_30/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp×
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMulÐ
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOpÀ
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMul_1¸
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/addÈ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOpÅ
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/BiasAdd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_30/split
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_1¡
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Reluµ
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_1ª
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/add_1
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_2
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Relu_1¹
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 

e
G__inference_dropout_49_layer_call_and_return_conditional_losses_1677186

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
ì 
ý
E__inference_dense_32_layer_call_and_return_conditional_losses_1677262

inputs4
!tensordot_readvariableop_resource:	´-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	´*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
¯?
Ó
while_body_1679446
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	ÐI
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	ÐG
3while_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_29_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_29/BiasAdd/ReadVariableOp¢(while/lstm_cell_29/MatMul/ReadVariableOp¢*while/lstm_cell_29/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	Ð*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp×
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMulÐ
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOpÀ
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMul_1¸
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/addÈ
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOpÅ
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/BiasAdd
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_29/split
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_1¡
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Reluµ
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_1ª
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/add_1
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_2
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Relu_1¹
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
èJ
Õ

lstm_31_while_body_1679047,
(lstm_31_while_lstm_31_while_loop_counter2
.lstm_31_while_lstm_31_while_maximum_iterations
lstm_31_while_placeholder
lstm_31_while_placeholder_1
lstm_31_while_placeholder_2
lstm_31_while_placeholder_3+
'lstm_31_while_lstm_31_strided_slice_1_0g
clstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0:
´ÐQ
=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0:
´ÐK
<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0:	Ð
lstm_31_while_identity
lstm_31_while_identity_1
lstm_31_while_identity_2
lstm_31_while_identity_3
lstm_31_while_identity_4
lstm_31_while_identity_5)
%lstm_31_while_lstm_31_strided_slice_1e
alstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensorM
9lstm_31_while_lstm_cell_31_matmul_readvariableop_resource:
´ÐO
;lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐI
:lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource:	Ð¢1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp¢0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp¢2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpÓ
?lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2A
?lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_31/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0lstm_31_while_placeholderHlstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype023
1lstm_31/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype022
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp÷
!lstm_31/while/lstm_cell_31/MatMulMatMul8lstm_31/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2#
!lstm_31/while/lstm_cell_31/MatMulè
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype024
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpà
#lstm_31/while/lstm_cell_31/MatMul_1MatMullstm_31_while_placeholder_2:lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2%
#lstm_31/while/lstm_cell_31/MatMul_1Ø
lstm_31/while/lstm_cell_31/addAddV2+lstm_31/while/lstm_cell_31/MatMul:product:0-lstm_31/while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2 
lstm_31/while/lstm_cell_31/addà
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype023
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOpå
"lstm_31/while/lstm_cell_31/BiasAddBiasAdd"lstm_31/while/lstm_cell_31/add:z:09lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2$
"lstm_31/while/lstm_cell_31/BiasAdd
*lstm_31/while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_31/while/lstm_cell_31/split/split_dim¯
 lstm_31/while/lstm_cell_31/splitSplit3lstm_31/while/lstm_cell_31/split/split_dim:output:0+lstm_31/while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2"
 lstm_31/while/lstm_cell_31/split±
"lstm_31/while/lstm_cell_31/SigmoidSigmoid)lstm_31/while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2$
"lstm_31/while/lstm_cell_31/Sigmoidµ
$lstm_31/while/lstm_cell_31/Sigmoid_1Sigmoid)lstm_31/while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_31/while/lstm_cell_31/Sigmoid_1Á
lstm_31/while/lstm_cell_31/mulMul(lstm_31/while/lstm_cell_31/Sigmoid_1:y:0lstm_31_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_31/while/lstm_cell_31/mul¨
lstm_31/while/lstm_cell_31/ReluRelu)lstm_31/while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
lstm_31/while/lstm_cell_31/ReluÕ
 lstm_31/while/lstm_cell_31/mul_1Mul&lstm_31/while/lstm_cell_31/Sigmoid:y:0-lstm_31/while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_31/while/lstm_cell_31/mul_1Ê
 lstm_31/while/lstm_cell_31/add_1AddV2"lstm_31/while/lstm_cell_31/mul:z:0$lstm_31/while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_31/while/lstm_cell_31/add_1µ
$lstm_31/while/lstm_cell_31/Sigmoid_2Sigmoid)lstm_31/while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_31/while/lstm_cell_31/Sigmoid_2§
!lstm_31/while/lstm_cell_31/Relu_1Relu$lstm_31/while/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!lstm_31/while/lstm_cell_31/Relu_1Ù
 lstm_31/while/lstm_cell_31/mul_2Mul(lstm_31/while/lstm_cell_31/Sigmoid_2:y:0/lstm_31/while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_31/while/lstm_cell_31/mul_2
2lstm_31/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_31_while_placeholder_1lstm_31_while_placeholder$lstm_31/while/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_31/while/TensorArrayV2Write/TensorListSetIteml
lstm_31/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_31/while/add/y
lstm_31/while/addAddV2lstm_31_while_placeholderlstm_31/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_31/while/addp
lstm_31/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_31/while/add_1/y
lstm_31/while/add_1AddV2(lstm_31_while_lstm_31_while_loop_counterlstm_31/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_31/while/add_1
lstm_31/while/IdentityIdentitylstm_31/while/add_1:z:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity¦
lstm_31/while/Identity_1Identity.lstm_31_while_lstm_31_while_maximum_iterations^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_1
lstm_31/while/Identity_2Identitylstm_31/while/add:z:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_2º
lstm_31/while/Identity_3IdentityBlstm_31/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_3®
lstm_31/while/Identity_4Identity$lstm_31/while/lstm_cell_31/mul_2:z:0^lstm_31/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/while/Identity_4®
lstm_31/while/Identity_5Identity$lstm_31/while/lstm_cell_31/add_1:z:0^lstm_31/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/while/Identity_5
lstm_31/while/NoOpNoOp2^lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp1^lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp3^lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_31/while/NoOp"9
lstm_31_while_identitylstm_31/while/Identity:output:0"=
lstm_31_while_identity_1!lstm_31/while/Identity_1:output:0"=
lstm_31_while_identity_2!lstm_31/while/Identity_2:output:0"=
lstm_31_while_identity_3!lstm_31/while/Identity_3:output:0"=
lstm_31_while_identity_4!lstm_31/while/Identity_4:output:0"=
lstm_31_while_identity_5!lstm_31/while/Identity_5:output:0"P
%lstm_31_while_lstm_31_strided_slice_1'lstm_31_while_lstm_31_strided_slice_1_0"z
:lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0"|
;lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0"x
9lstm_31_while_lstm_cell_31_matmul_readvariableop_resource;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0"È
alstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensorclstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2f
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp2d
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp2h
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
ÃU

D__inference_lstm_30_layer_call_and_return_conditional_losses_1680459

inputs?
+lstm_cell_30_matmul_readvariableop_resource:
´ÐA
-lstm_cell_30_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_30_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_30/BiasAdd/ReadVariableOp¢"lstm_cell_30/MatMul/ReadVariableOp¢$lstm_cell_30/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp­
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul¼
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp©
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul_1 
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/add´
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp­
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim÷
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_30/split
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_1
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_1
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/add_1
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu_1¡
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1680375*
condR
while_cond_1680374*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Þ
¹
)__inference_lstm_29_layer_call_fn_1679222
inputs_0
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_16752682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Þ
È
while_cond_1677623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1677623___redundant_placeholder05
1while_while_cond_1677623___redundant_placeholder15
1while_while_cond_1677623___redundant_placeholder25
1while_while_cond_1677623___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
Üî
½
J__inference_sequential_13_layer_call_and_return_conditional_losses_1678694

inputsF
3lstm_29_lstm_cell_29_matmul_readvariableop_resource:	ÐI
5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐC
4lstm_29_lstm_cell_29_biasadd_readvariableop_resource:	ÐG
3lstm_30_lstm_cell_30_matmul_readvariableop_resource:
´ÐI
5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐC
4lstm_30_lstm_cell_30_biasadd_readvariableop_resource:	ÐG
3lstm_31_lstm_cell_31_matmul_readvariableop_resource:
´ÐI
5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐC
4lstm_31_lstm_cell_31_biasadd_readvariableop_resource:	Ð>
*dense_31_tensordot_readvariableop_resource:
´´7
(dense_31_biasadd_readvariableop_resource:	´=
*dense_32_tensordot_readvariableop_resource:	´6
(dense_32_biasadd_readvariableop_resource:
identity¢dense_31/BiasAdd/ReadVariableOp¢!dense_31/Tensordot/ReadVariableOp¢dense_32/BiasAdd/ReadVariableOp¢!dense_32/Tensordot/ReadVariableOp¢+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp¢*lstm_29/lstm_cell_29/MatMul/ReadVariableOp¢,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp¢lstm_29/while¢+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp¢*lstm_30/lstm_cell_30/MatMul/ReadVariableOp¢,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp¢lstm_30/while¢+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp¢*lstm_31/lstm_cell_31/MatMul/ReadVariableOp¢,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp¢lstm_31/whileT
lstm_29/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_29/Shape
lstm_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice/stack
lstm_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_29/strided_slice/stack_1
lstm_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_29/strided_slice/stack_2
lstm_29/strided_sliceStridedSlicelstm_29/Shape:output:0$lstm_29/strided_slice/stack:output:0&lstm_29/strided_slice/stack_1:output:0&lstm_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_29/strided_slices
lstm_29/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_29/zeros/packed/1£
lstm_29/zeros/packedPacklstm_29/strided_slice:output:0lstm_29/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_29/zeros/packedo
lstm_29/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/zeros/Const
lstm_29/zerosFilllstm_29/zeros/packed:output:0lstm_29/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/zerosw
lstm_29/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_29/zeros_1/packed/1©
lstm_29/zeros_1/packedPacklstm_29/strided_slice:output:0!lstm_29/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_29/zeros_1/packeds
lstm_29/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/zeros_1/Const
lstm_29/zeros_1Filllstm_29/zeros_1/packed:output:0lstm_29/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/zeros_1
lstm_29/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_29/transpose/perm
lstm_29/transpose	Transposeinputslstm_29/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_29/transposeg
lstm_29/Shape_1Shapelstm_29/transpose:y:0*
T0*
_output_shapes
:2
lstm_29/Shape_1
lstm_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice_1/stack
lstm_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_1/stack_1
lstm_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_1/stack_2
lstm_29/strided_slice_1StridedSlicelstm_29/Shape_1:output:0&lstm_29/strided_slice_1/stack:output:0(lstm_29/strided_slice_1/stack_1:output:0(lstm_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_29/strided_slice_1
#lstm_29/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_29/TensorArrayV2/element_shapeÒ
lstm_29/TensorArrayV2TensorListReserve,lstm_29/TensorArrayV2/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_29/TensorArrayV2Ï
=lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_29/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_29/transpose:y:0Flstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_29/TensorArrayUnstack/TensorListFromTensor
lstm_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice_2/stack
lstm_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_2/stack_1
lstm_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_2/stack_2¬
lstm_29/strided_slice_2StridedSlicelstm_29/transpose:y:0&lstm_29/strided_slice_2/stack:output:0(lstm_29/strided_slice_2/stack_1:output:0(lstm_29/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_29/strided_slice_2Í
*lstm_29/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3lstm_29_lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02,
*lstm_29/lstm_cell_29/MatMul/ReadVariableOpÍ
lstm_29/lstm_cell_29/MatMulMatMul lstm_29/strided_slice_2:output:02lstm_29/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_29/lstm_cell_29/MatMulÔ
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02.
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOpÉ
lstm_29/lstm_cell_29/MatMul_1MatMullstm_29/zeros:output:04lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_29/lstm_cell_29/MatMul_1À
lstm_29/lstm_cell_29/addAddV2%lstm_29/lstm_cell_29/MatMul:product:0'lstm_29/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_29/lstm_cell_29/addÌ
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02-
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOpÍ
lstm_29/lstm_cell_29/BiasAddBiasAddlstm_29/lstm_cell_29/add:z:03lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_29/lstm_cell_29/BiasAdd
$lstm_29/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_29/lstm_cell_29/split/split_dim
lstm_29/lstm_cell_29/splitSplit-lstm_29/lstm_cell_29/split/split_dim:output:0%lstm_29/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_29/lstm_cell_29/split
lstm_29/lstm_cell_29/SigmoidSigmoid#lstm_29/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/Sigmoid£
lstm_29/lstm_cell_29/Sigmoid_1Sigmoid#lstm_29/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_29/lstm_cell_29/Sigmoid_1¬
lstm_29/lstm_cell_29/mulMul"lstm_29/lstm_cell_29/Sigmoid_1:y:0lstm_29/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/mul
lstm_29/lstm_cell_29/ReluRelu#lstm_29/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/Relu½
lstm_29/lstm_cell_29/mul_1Mul lstm_29/lstm_cell_29/Sigmoid:y:0'lstm_29/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/mul_1²
lstm_29/lstm_cell_29/add_1AddV2lstm_29/lstm_cell_29/mul:z:0lstm_29/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/add_1£
lstm_29/lstm_cell_29/Sigmoid_2Sigmoid#lstm_29/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_29/lstm_cell_29/Sigmoid_2
lstm_29/lstm_cell_29/Relu_1Relulstm_29/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/Relu_1Á
lstm_29/lstm_cell_29/mul_2Mul"lstm_29/lstm_cell_29/Sigmoid_2:y:0)lstm_29/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/mul_2
%lstm_29/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2'
%lstm_29/TensorArrayV2_1/element_shapeØ
lstm_29/TensorArrayV2_1TensorListReserve.lstm_29/TensorArrayV2_1/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_29/TensorArrayV2_1^
lstm_29/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_29/time
 lstm_29/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_29/while/maximum_iterationsz
lstm_29/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_29/while/loop_counter
lstm_29/whileWhile#lstm_29/while/loop_counter:output:0)lstm_29/while/maximum_iterations:output:0lstm_29/time:output:0 lstm_29/TensorArrayV2_1:handle:0lstm_29/zeros:output:0lstm_29/zeros_1:output:0 lstm_29/strided_slice_1:output:0?lstm_29/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_29_lstm_cell_29_matmul_readvariableop_resource5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource4lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_29_while_body_1678275*&
condR
lstm_29_while_cond_1678274*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
lstm_29/whileÅ
8lstm_29/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2:
8lstm_29/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_29/TensorArrayV2Stack/TensorListStackTensorListStacklstm_29/while:output:3Alstm_29/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02,
*lstm_29/TensorArrayV2Stack/TensorListStack
lstm_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_29/strided_slice_3/stack
lstm_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_29/strided_slice_3/stack_1
lstm_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_3/stack_2Ë
lstm_29/strided_slice_3StridedSlice3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_29/strided_slice_3/stack:output:0(lstm_29/strided_slice_3/stack_1:output:0(lstm_29/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
lstm_29/strided_slice_3
lstm_29/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_29/transpose_1/permÆ
lstm_29/transpose_1	Transpose3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_29/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/transpose_1v
lstm_29/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/runtime
dropout_47/IdentityIdentitylstm_29/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_47/Identityj
lstm_30/ShapeShapedropout_47/Identity:output:0*
T0*
_output_shapes
:2
lstm_30/Shape
lstm_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice/stack
lstm_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_30/strided_slice/stack_1
lstm_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_30/strided_slice/stack_2
lstm_30/strided_sliceStridedSlicelstm_30/Shape:output:0$lstm_30/strided_slice/stack:output:0&lstm_30/strided_slice/stack_1:output:0&lstm_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_30/strided_slices
lstm_30/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_30/zeros/packed/1£
lstm_30/zeros/packedPacklstm_30/strided_slice:output:0lstm_30/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_30/zeros/packedo
lstm_30/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/zeros/Const
lstm_30/zerosFilllstm_30/zeros/packed:output:0lstm_30/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/zerosw
lstm_30/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_30/zeros_1/packed/1©
lstm_30/zeros_1/packedPacklstm_30/strided_slice:output:0!lstm_30/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_30/zeros_1/packeds
lstm_30/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/zeros_1/Const
lstm_30/zeros_1Filllstm_30/zeros_1/packed:output:0lstm_30/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/zeros_1
lstm_30/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_30/transpose/perm©
lstm_30/transpose	Transposedropout_47/Identity:output:0lstm_30/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/transposeg
lstm_30/Shape_1Shapelstm_30/transpose:y:0*
T0*
_output_shapes
:2
lstm_30/Shape_1
lstm_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice_1/stack
lstm_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_1/stack_1
lstm_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_1/stack_2
lstm_30/strided_slice_1StridedSlicelstm_30/Shape_1:output:0&lstm_30/strided_slice_1/stack:output:0(lstm_30/strided_slice_1/stack_1:output:0(lstm_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_30/strided_slice_1
#lstm_30/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_30/TensorArrayV2/element_shapeÒ
lstm_30/TensorArrayV2TensorListReserve,lstm_30/TensorArrayV2/element_shape:output:0 lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_30/TensorArrayV2Ï
=lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2?
=lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_30/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_30/transpose:y:0Flstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_30/TensorArrayUnstack/TensorListFromTensor
lstm_30/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice_2/stack
lstm_30/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_2/stack_1
lstm_30/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_2/stack_2­
lstm_30/strided_slice_2StridedSlicelstm_30/transpose:y:0&lstm_30/strided_slice_2/stack:output:0(lstm_30/strided_slice_2/stack_1:output:0(lstm_30/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
lstm_30/strided_slice_2Î
*lstm_30/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3lstm_30_lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02,
*lstm_30/lstm_cell_30/MatMul/ReadVariableOpÍ
lstm_30/lstm_cell_30/MatMulMatMul lstm_30/strided_slice_2:output:02lstm_30/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_30/lstm_cell_30/MatMulÔ
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02.
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOpÉ
lstm_30/lstm_cell_30/MatMul_1MatMullstm_30/zeros:output:04lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_30/lstm_cell_30/MatMul_1À
lstm_30/lstm_cell_30/addAddV2%lstm_30/lstm_cell_30/MatMul:product:0'lstm_30/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_30/lstm_cell_30/addÌ
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02-
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOpÍ
lstm_30/lstm_cell_30/BiasAddBiasAddlstm_30/lstm_cell_30/add:z:03lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_30/lstm_cell_30/BiasAdd
$lstm_30/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_30/lstm_cell_30/split/split_dim
lstm_30/lstm_cell_30/splitSplit-lstm_30/lstm_cell_30/split/split_dim:output:0%lstm_30/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_30/lstm_cell_30/split
lstm_30/lstm_cell_30/SigmoidSigmoid#lstm_30/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/Sigmoid£
lstm_30/lstm_cell_30/Sigmoid_1Sigmoid#lstm_30/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_30/lstm_cell_30/Sigmoid_1¬
lstm_30/lstm_cell_30/mulMul"lstm_30/lstm_cell_30/Sigmoid_1:y:0lstm_30/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/mul
lstm_30/lstm_cell_30/ReluRelu#lstm_30/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/Relu½
lstm_30/lstm_cell_30/mul_1Mul lstm_30/lstm_cell_30/Sigmoid:y:0'lstm_30/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/mul_1²
lstm_30/lstm_cell_30/add_1AddV2lstm_30/lstm_cell_30/mul:z:0lstm_30/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/add_1£
lstm_30/lstm_cell_30/Sigmoid_2Sigmoid#lstm_30/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_30/lstm_cell_30/Sigmoid_2
lstm_30/lstm_cell_30/Relu_1Relulstm_30/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/Relu_1Á
lstm_30/lstm_cell_30/mul_2Mul"lstm_30/lstm_cell_30/Sigmoid_2:y:0)lstm_30/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/mul_2
%lstm_30/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2'
%lstm_30/TensorArrayV2_1/element_shapeØ
lstm_30/TensorArrayV2_1TensorListReserve.lstm_30/TensorArrayV2_1/element_shape:output:0 lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_30/TensorArrayV2_1^
lstm_30/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_30/time
 lstm_30/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_30/while/maximum_iterationsz
lstm_30/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_30/while/loop_counter
lstm_30/whileWhile#lstm_30/while/loop_counter:output:0)lstm_30/while/maximum_iterations:output:0lstm_30/time:output:0 lstm_30/TensorArrayV2_1:handle:0lstm_30/zeros:output:0lstm_30/zeros_1:output:0 lstm_30/strided_slice_1:output:0?lstm_30/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_30_lstm_cell_30_matmul_readvariableop_resource5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource4lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_30_while_body_1678415*&
condR
lstm_30_while_cond_1678414*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
lstm_30/whileÅ
8lstm_30/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2:
8lstm_30/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_30/TensorArrayV2Stack/TensorListStackTensorListStacklstm_30/while:output:3Alstm_30/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02,
*lstm_30/TensorArrayV2Stack/TensorListStack
lstm_30/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_30/strided_slice_3/stack
lstm_30/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_30/strided_slice_3/stack_1
lstm_30/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_3/stack_2Ë
lstm_30/strided_slice_3StridedSlice3lstm_30/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_30/strided_slice_3/stack:output:0(lstm_30/strided_slice_3/stack_1:output:0(lstm_30/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
lstm_30/strided_slice_3
lstm_30/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_30/transpose_1/permÆ
lstm_30/transpose_1	Transpose3lstm_30/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_30/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/transpose_1v
lstm_30/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/runtime
dropout_48/IdentityIdentitylstm_30/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_48/Identityj
lstm_31/ShapeShapedropout_48/Identity:output:0*
T0*
_output_shapes
:2
lstm_31/Shape
lstm_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice/stack
lstm_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_31/strided_slice/stack_1
lstm_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_31/strided_slice/stack_2
lstm_31/strided_sliceStridedSlicelstm_31/Shape:output:0$lstm_31/strided_slice/stack:output:0&lstm_31/strided_slice/stack_1:output:0&lstm_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_31/strided_slices
lstm_31/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_31/zeros/packed/1£
lstm_31/zeros/packedPacklstm_31/strided_slice:output:0lstm_31/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_31/zeros/packedo
lstm_31/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/zeros/Const
lstm_31/zerosFilllstm_31/zeros/packed:output:0lstm_31/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/zerosw
lstm_31/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_31/zeros_1/packed/1©
lstm_31/zeros_1/packedPacklstm_31/strided_slice:output:0!lstm_31/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_31/zeros_1/packeds
lstm_31/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/zeros_1/Const
lstm_31/zeros_1Filllstm_31/zeros_1/packed:output:0lstm_31/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/zeros_1
lstm_31/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_31/transpose/perm©
lstm_31/transpose	Transposedropout_48/Identity:output:0lstm_31/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/transposeg
lstm_31/Shape_1Shapelstm_31/transpose:y:0*
T0*
_output_shapes
:2
lstm_31/Shape_1
lstm_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice_1/stack
lstm_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_1/stack_1
lstm_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_1/stack_2
lstm_31/strided_slice_1StridedSlicelstm_31/Shape_1:output:0&lstm_31/strided_slice_1/stack:output:0(lstm_31/strided_slice_1/stack_1:output:0(lstm_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_31/strided_slice_1
#lstm_31/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_31/TensorArrayV2/element_shapeÒ
lstm_31/TensorArrayV2TensorListReserve,lstm_31/TensorArrayV2/element_shape:output:0 lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_31/TensorArrayV2Ï
=lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2?
=lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_31/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_31/transpose:y:0Flstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_31/TensorArrayUnstack/TensorListFromTensor
lstm_31/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice_2/stack
lstm_31/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_2/stack_1
lstm_31/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_2/stack_2­
lstm_31/strided_slice_2StridedSlicelstm_31/transpose:y:0&lstm_31/strided_slice_2/stack:output:0(lstm_31/strided_slice_2/stack_1:output:0(lstm_31/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
lstm_31/strided_slice_2Î
*lstm_31/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3lstm_31_lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02,
*lstm_31/lstm_cell_31/MatMul/ReadVariableOpÍ
lstm_31/lstm_cell_31/MatMulMatMul lstm_31/strided_slice_2:output:02lstm_31/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_31/lstm_cell_31/MatMulÔ
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02.
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOpÉ
lstm_31/lstm_cell_31/MatMul_1MatMullstm_31/zeros:output:04lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_31/lstm_cell_31/MatMul_1À
lstm_31/lstm_cell_31/addAddV2%lstm_31/lstm_cell_31/MatMul:product:0'lstm_31/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_31/lstm_cell_31/addÌ
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02-
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOpÍ
lstm_31/lstm_cell_31/BiasAddBiasAddlstm_31/lstm_cell_31/add:z:03lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_31/lstm_cell_31/BiasAdd
$lstm_31/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_31/lstm_cell_31/split/split_dim
lstm_31/lstm_cell_31/splitSplit-lstm_31/lstm_cell_31/split/split_dim:output:0%lstm_31/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_31/lstm_cell_31/split
lstm_31/lstm_cell_31/SigmoidSigmoid#lstm_31/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/Sigmoid£
lstm_31/lstm_cell_31/Sigmoid_1Sigmoid#lstm_31/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_31/lstm_cell_31/Sigmoid_1¬
lstm_31/lstm_cell_31/mulMul"lstm_31/lstm_cell_31/Sigmoid_1:y:0lstm_31/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/mul
lstm_31/lstm_cell_31/ReluRelu#lstm_31/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/Relu½
lstm_31/lstm_cell_31/mul_1Mul lstm_31/lstm_cell_31/Sigmoid:y:0'lstm_31/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/mul_1²
lstm_31/lstm_cell_31/add_1AddV2lstm_31/lstm_cell_31/mul:z:0lstm_31/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/add_1£
lstm_31/lstm_cell_31/Sigmoid_2Sigmoid#lstm_31/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_31/lstm_cell_31/Sigmoid_2
lstm_31/lstm_cell_31/Relu_1Relulstm_31/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/Relu_1Á
lstm_31/lstm_cell_31/mul_2Mul"lstm_31/lstm_cell_31/Sigmoid_2:y:0)lstm_31/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/mul_2
%lstm_31/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2'
%lstm_31/TensorArrayV2_1/element_shapeØ
lstm_31/TensorArrayV2_1TensorListReserve.lstm_31/TensorArrayV2_1/element_shape:output:0 lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_31/TensorArrayV2_1^
lstm_31/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_31/time
 lstm_31/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_31/while/maximum_iterationsz
lstm_31/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_31/while/loop_counter
lstm_31/whileWhile#lstm_31/while/loop_counter:output:0)lstm_31/while/maximum_iterations:output:0lstm_31/time:output:0 lstm_31/TensorArrayV2_1:handle:0lstm_31/zeros:output:0lstm_31/zeros_1:output:0 lstm_31/strided_slice_1:output:0?lstm_31/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_31_lstm_cell_31_matmul_readvariableop_resource5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource4lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_31_while_body_1678555*&
condR
lstm_31_while_cond_1678554*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
lstm_31/whileÅ
8lstm_31/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2:
8lstm_31/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_31/TensorArrayV2Stack/TensorListStackTensorListStacklstm_31/while:output:3Alstm_31/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02,
*lstm_31/TensorArrayV2Stack/TensorListStack
lstm_31/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_31/strided_slice_3/stack
lstm_31/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_31/strided_slice_3/stack_1
lstm_31/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_3/stack_2Ë
lstm_31/strided_slice_3StridedSlice3lstm_31/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_31/strided_slice_3/stack:output:0(lstm_31/strided_slice_3/stack_1:output:0(lstm_31/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
lstm_31/strided_slice_3
lstm_31/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_31/transpose_1/permÆ
lstm_31/transpose_1	Transpose3lstm_31/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_31/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/transpose_1v
lstm_31/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/runtime
dropout_49/IdentityIdentitylstm_31/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_49/Identity³
!dense_31/Tensordot/ReadVariableOpReadVariableOp*dense_31_tensordot_readvariableop_resource* 
_output_shapes
:
´´*
dtype02#
!dense_31/Tensordot/ReadVariableOp|
dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_31/Tensordot/axes
dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_31/Tensordot/free
dense_31/Tensordot/ShapeShapedropout_49/Identity:output:0*
T0*
_output_shapes
:2
dense_31/Tensordot/Shape
 dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/GatherV2/axisþ
dense_31/Tensordot/GatherV2GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/free:output:0)dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2
"dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_31/Tensordot/GatherV2_1/axis
dense_31/Tensordot/GatherV2_1GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/axes:output:0+dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2_1~
dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const¤
dense_31/Tensordot/ProdProd$dense_31/Tensordot/GatherV2:output:0!dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod
dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const_1¬
dense_31/Tensordot/Prod_1Prod&dense_31/Tensordot/GatherV2_1:output:0#dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod_1
dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_31/Tensordot/concat/axisÝ
dense_31/Tensordot/concatConcatV2 dense_31/Tensordot/free:output:0 dense_31/Tensordot/axes:output:0'dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat°
dense_31/Tensordot/stackPack dense_31/Tensordot/Prod:output:0"dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/stackÂ
dense_31/Tensordot/transpose	Transposedropout_49/Identity:output:0"dense_31/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_31/Tensordot/transposeÃ
dense_31/Tensordot/ReshapeReshape dense_31/Tensordot/transpose:y:0!dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_31/Tensordot/ReshapeÃ
dense_31/Tensordot/MatMulMatMul#dense_31/Tensordot/Reshape:output:0)dense_31/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_31/Tensordot/MatMul
dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:´2
dense_31/Tensordot/Const_2
 dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/concat_1/axisê
dense_31/Tensordot/concat_1ConcatV2$dense_31/Tensordot/GatherV2:output:0#dense_31/Tensordot/Const_2:output:0)dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat_1µ
dense_31/TensordotReshape#dense_31/Tensordot/MatMul:product:0$dense_31/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_31/Tensordot¨
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:´*
dtype02!
dense_31/BiasAdd/ReadVariableOp¬
dense_31/BiasAddBiasAdddense_31/Tensordot:output:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_31/BiasAddx
dense_31/ReluReludense_31/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_31/Relu
dropout_50/IdentityIdentitydense_31/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_50/Identity²
!dense_32/Tensordot/ReadVariableOpReadVariableOp*dense_32_tensordot_readvariableop_resource*
_output_shapes
:	´*
dtype02#
!dense_32/Tensordot/ReadVariableOp|
dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_32/Tensordot/axes
dense_32/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_32/Tensordot/free
dense_32/Tensordot/ShapeShapedropout_50/Identity:output:0*
T0*
_output_shapes
:2
dense_32/Tensordot/Shape
 dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/GatherV2/axisþ
dense_32/Tensordot/GatherV2GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/free:output:0)dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2
"dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_32/Tensordot/GatherV2_1/axis
dense_32/Tensordot/GatherV2_1GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/axes:output:0+dense_32/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2_1~
dense_32/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const¤
dense_32/Tensordot/ProdProd$dense_32/Tensordot/GatherV2:output:0!dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod
dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_1¬
dense_32/Tensordot/Prod_1Prod&dense_32/Tensordot/GatherV2_1:output:0#dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod_1
dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_32/Tensordot/concat/axisÝ
dense_32/Tensordot/concatConcatV2 dense_32/Tensordot/free:output:0 dense_32/Tensordot/axes:output:0'dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat°
dense_32/Tensordot/stackPack dense_32/Tensordot/Prod:output:0"dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/stackÂ
dense_32/Tensordot/transpose	Transposedropout_50/Identity:output:0"dense_32/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_32/Tensordot/transposeÃ
dense_32/Tensordot/ReshapeReshape dense_32/Tensordot/transpose:y:0!dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_32/Tensordot/ReshapeÂ
dense_32/Tensordot/MatMulMatMul#dense_32/Tensordot/Reshape:output:0)dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_32/Tensordot/MatMul
dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_32/Tensordot/Const_2
 dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/concat_1/axisê
dense_32/Tensordot/concat_1ConcatV2$dense_32/Tensordot/GatherV2:output:0#dense_32/Tensordot/Const_2:output:0)dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat_1´
dense_32/TensordotReshape#dense_32/Tensordot/MatMul:product:0$dense_32/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_32/Tensordot§
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_32/BiasAdd/ReadVariableOp«
dense_32/BiasAddBiasAdddense_32/Tensordot:output:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_32/BiasAddx
IdentityIdentitydense_32/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¨
NoOpNoOp ^dense_31/BiasAdd/ReadVariableOp"^dense_31/Tensordot/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp"^dense_32/Tensordot/ReadVariableOp,^lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp+^lstm_29/lstm_cell_29/MatMul/ReadVariableOp-^lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp^lstm_29/while,^lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp+^lstm_30/lstm_cell_30/MatMul/ReadVariableOp-^lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp^lstm_30/while,^lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp+^lstm_31/lstm_cell_31/MatMul/ReadVariableOp-^lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp^lstm_31/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2F
!dense_31/Tensordot/ReadVariableOp!dense_31/Tensordot/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2F
!dense_32/Tensordot/ReadVariableOp!dense_32/Tensordot/ReadVariableOp2Z
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp2X
*lstm_29/lstm_cell_29/MatMul/ReadVariableOp*lstm_29/lstm_cell_29/MatMul/ReadVariableOp2\
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp2
lstm_29/whilelstm_29/while2Z
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp2X
*lstm_30/lstm_cell_30/MatMul/ReadVariableOp*lstm_30/lstm_cell_30/MatMul/ReadVariableOp2\
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp2
lstm_30/whilelstm_30/while2Z
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp2X
*lstm_31/lstm_cell_31/MatMul/ReadVariableOp*lstm_31/lstm_cell_31/MatMul/ReadVariableOp2\
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp2
lstm_31/whilelstm_31/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
e
,__inference_dropout_47_layer_call_fn_1679826

inputs
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_16777372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
¹
e
,__inference_dropout_48_layer_call_fn_1680469

inputs
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_16775492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs

e
G__inference_dropout_47_layer_call_and_return_conditional_losses_1676872

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Þ
È
while_cond_1679445
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1679445___redundant_placeholder05
1while_while_cond_1679445___redundant_placeholder15
1while_while_cond_1679445___redundant_placeholder25
1while_while_cond_1679445___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:

æ
/__inference_sequential_13_layer_call_fn_1677298
lstm_29_input
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
	unknown_2:
´Ð
	unknown_3:
´Ð
	unknown_4:	Ð
	unknown_5:
´Ð
	unknown_6:
´Ð
	unknown_7:	Ð
	unknown_8:
´´
	unknown_9:	´

unknown_10:	´

unknown_11:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_29_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_16772692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_29_input
ÿ?

D__inference_lstm_31_layer_call_and_return_conditional_losses_1676262

inputs(
lstm_cell_31_1676180:
´Ð(
lstm_cell_31_1676182:
´Ð#
lstm_cell_31_1676184:	Ð
identity¢$lstm_cell_31/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¤
$lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_31_1676180lstm_cell_31_1676182lstm_cell_31_1676184*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_16761792&
$lstm_cell_31/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÉ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_31_1676180lstm_cell_31_1676182lstm_cell_31_1676184*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1676193*
condR
while_cond_1676192*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identity}
NoOpNoOp%^lstm_cell_31/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 2L
$lstm_cell_31/StatefulPartitionedCall$lstm_cell_31/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs


I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1674983

inputs

states
states_11
matmul_readvariableop_resource:	Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates
ãÑ
Ö 
#__inference__traced_restore_1681850
file_prefix4
 assignvariableop_dense_31_kernel:
´´/
 assignvariableop_1_dense_31_bias:	´5
"assignvariableop_2_dense_32_kernel:	´.
 assignvariableop_3_dense_32_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: A
.assignvariableop_9_lstm_29_lstm_cell_29_kernel:	ÐM
9assignvariableop_10_lstm_29_lstm_cell_29_recurrent_kernel:
´Ð<
-assignvariableop_11_lstm_29_lstm_cell_29_bias:	ÐC
/assignvariableop_12_lstm_30_lstm_cell_30_kernel:
´ÐM
9assignvariableop_13_lstm_30_lstm_cell_30_recurrent_kernel:
´Ð<
-assignvariableop_14_lstm_30_lstm_cell_30_bias:	ÐC
/assignvariableop_15_lstm_31_lstm_cell_31_kernel:
´ÐM
9assignvariableop_16_lstm_31_lstm_cell_31_recurrent_kernel:
´Ð<
-assignvariableop_17_lstm_31_lstm_cell_31_bias:	Ð#
assignvariableop_18_total: #
assignvariableop_19_count: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: >
*assignvariableop_22_adam_dense_31_kernel_m:
´´7
(assignvariableop_23_adam_dense_31_bias_m:	´=
*assignvariableop_24_adam_dense_32_kernel_m:	´6
(assignvariableop_25_adam_dense_32_bias_m:I
6assignvariableop_26_adam_lstm_29_lstm_cell_29_kernel_m:	ÐT
@assignvariableop_27_adam_lstm_29_lstm_cell_29_recurrent_kernel_m:
´ÐC
4assignvariableop_28_adam_lstm_29_lstm_cell_29_bias_m:	ÐJ
6assignvariableop_29_adam_lstm_30_lstm_cell_30_kernel_m:
´ÐT
@assignvariableop_30_adam_lstm_30_lstm_cell_30_recurrent_kernel_m:
´ÐC
4assignvariableop_31_adam_lstm_30_lstm_cell_30_bias_m:	ÐJ
6assignvariableop_32_adam_lstm_31_lstm_cell_31_kernel_m:
´ÐT
@assignvariableop_33_adam_lstm_31_lstm_cell_31_recurrent_kernel_m:
´ÐC
4assignvariableop_34_adam_lstm_31_lstm_cell_31_bias_m:	Ð>
*assignvariableop_35_adam_dense_31_kernel_v:
´´7
(assignvariableop_36_adam_dense_31_bias_v:	´=
*assignvariableop_37_adam_dense_32_kernel_v:	´6
(assignvariableop_38_adam_dense_32_bias_v:I
6assignvariableop_39_adam_lstm_29_lstm_cell_29_kernel_v:	ÐT
@assignvariableop_40_adam_lstm_29_lstm_cell_29_recurrent_kernel_v:
´ÐC
4assignvariableop_41_adam_lstm_29_lstm_cell_29_bias_v:	ÐJ
6assignvariableop_42_adam_lstm_30_lstm_cell_30_kernel_v:
´ÐT
@assignvariableop_43_adam_lstm_30_lstm_cell_30_recurrent_kernel_v:
´ÐC
4assignvariableop_44_adam_lstm_30_lstm_cell_30_bias_v:	ÐJ
6assignvariableop_45_adam_lstm_31_lstm_cell_31_kernel_v:
´ÐT
@assignvariableop_46_adam_lstm_31_lstm_cell_31_recurrent_kernel_v:
´ÐC
4assignvariableop_47_adam_lstm_31_lstm_cell_31_bias_v:	Ð
identity_49¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*
valueB1B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesð
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices£
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ú
_output_shapesÇ
Ä:::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes5
321	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_31_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_31_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_32_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_32_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4¡
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ª
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_29_lstm_cell_29_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Á
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_29_lstm_cell_29_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11µ
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_29_lstm_cell_29_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_lstm_30_lstm_cell_30_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Á
AssignVariableOp_13AssignVariableOp9assignvariableop_13_lstm_30_lstm_cell_30_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14µ
AssignVariableOp_14AssignVariableOp-assignvariableop_14_lstm_30_lstm_cell_30_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15·
AssignVariableOp_15AssignVariableOp/assignvariableop_15_lstm_31_lstm_cell_31_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Á
AssignVariableOp_16AssignVariableOp9assignvariableop_16_lstm_31_lstm_cell_31_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17µ
AssignVariableOp_17AssignVariableOp-assignvariableop_17_lstm_31_lstm_cell_31_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¡
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22²
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_31_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23°
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_31_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_32_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25°
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_32_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¾
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_lstm_29_lstm_cell_29_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27È
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_lstm_29_lstm_cell_29_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¼
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_lstm_29_lstm_cell_29_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¾
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_lstm_30_lstm_cell_30_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30È
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_lstm_30_lstm_cell_30_recurrent_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¼
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_lstm_30_lstm_cell_30_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¾
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_lstm_31_lstm_cell_31_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33È
AssignVariableOp_33AssignVariableOp@assignvariableop_33_adam_lstm_31_lstm_cell_31_recurrent_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¼
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_lstm_31_lstm_cell_31_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_31_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36°
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_31_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_32_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_32_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¾
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_lstm_29_lstm_cell_29_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40È
AssignVariableOp_40AssignVariableOp@assignvariableop_40_adam_lstm_29_lstm_cell_29_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¼
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_lstm_29_lstm_cell_29_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¾
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_lstm_30_lstm_cell_30_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43È
AssignVariableOp_43AssignVariableOp@assignvariableop_43_adam_lstm_30_lstm_cell_30_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¼
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_lstm_30_lstm_cell_30_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¾
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_lstm_31_lstm_cell_31_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46È
AssignVariableOp_46AssignVariableOp@assignvariableop_46_adam_lstm_31_lstm_cell_31_recurrent_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¼
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_lstm_31_lstm_cell_31_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_479
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpþ
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_48f
Identity_49IdentityIdentity_48:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_49æ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_49Identity_49:output:0*u
_input_shapesd
b: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
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
Þ
È
while_cond_1676394
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1676394___redundant_placeholder05
1while_while_cond_1676394___redundant_placeholder15
1while_while_cond_1676394___redundant_placeholder25
1while_while_cond_1676394___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:

e
G__inference_dropout_50_layer_call_and_return_conditional_losses_1677230

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
³?
Õ
while_body_1680089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_30_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_30/BiasAdd/ReadVariableOp¢(while/lstm_cell_30/MatMul/ReadVariableOp¢*while/lstm_cell_30/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp×
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMulÐ
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOpÀ
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMul_1¸
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/addÈ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOpÅ
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/BiasAdd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_30/split
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_1¡
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Reluµ
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_1ª
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/add_1
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_2
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Relu_1¹
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
ã^

(sequential_13_lstm_29_while_body_1674497H
Dsequential_13_lstm_29_while_sequential_13_lstm_29_while_loop_counterN
Jsequential_13_lstm_29_while_sequential_13_lstm_29_while_maximum_iterations+
'sequential_13_lstm_29_while_placeholder-
)sequential_13_lstm_29_while_placeholder_1-
)sequential_13_lstm_29_while_placeholder_2-
)sequential_13_lstm_29_while_placeholder_3G
Csequential_13_lstm_29_while_sequential_13_lstm_29_strided_slice_1_0
sequential_13_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_29_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_13_lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0:	Ð_
Ksequential_13_lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0:
´ÐY
Jsequential_13_lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0:	Ð(
$sequential_13_lstm_29_while_identity*
&sequential_13_lstm_29_while_identity_1*
&sequential_13_lstm_29_while_identity_2*
&sequential_13_lstm_29_while_identity_3*
&sequential_13_lstm_29_while_identity_4*
&sequential_13_lstm_29_while_identity_5E
Asequential_13_lstm_29_while_sequential_13_lstm_29_strided_slice_1
}sequential_13_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_29_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_13_lstm_29_while_lstm_cell_29_matmul_readvariableop_resource:	Ð]
Isequential_13_lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐW
Hsequential_13_lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource:	Ð¢?sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp¢>sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp¢@sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpï
Msequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_13_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_29_tensorarrayunstack_tensorlistfromtensor_0'sequential_13_lstm_29_while_placeholderVsequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItem
>sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOpIsequential_13_lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	Ð*
dtype02@
>sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp¯
/sequential_13/lstm_29/while/lstm_cell_29/MatMulMatMulFsequential_13/lstm_29/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ21
/sequential_13/lstm_29/while/lstm_cell_29/MatMul
@sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOpKsequential_13_lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02B
@sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp
1sequential_13/lstm_29/while/lstm_cell_29/MatMul_1MatMul)sequential_13_lstm_29_while_placeholder_2Hsequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ23
1sequential_13/lstm_29/while/lstm_cell_29/MatMul_1
,sequential_13/lstm_29/while/lstm_cell_29/addAddV29sequential_13/lstm_29/while/lstm_cell_29/MatMul:product:0;sequential_13/lstm_29/while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2.
,sequential_13/lstm_29/while/lstm_cell_29/add
?sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOpJsequential_13_lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02A
?sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp
0sequential_13/lstm_29/while/lstm_cell_29/BiasAddBiasAdd0sequential_13/lstm_29/while/lstm_cell_29/add:z:0Gsequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ22
0sequential_13/lstm_29/while/lstm_cell_29/BiasAdd¶
8sequential_13/lstm_29/while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_13/lstm_29/while/lstm_cell_29/split/split_dimç
.sequential_13/lstm_29/while/lstm_cell_29/splitSplitAsequential_13/lstm_29/while/lstm_cell_29/split/split_dim:output:09sequential_13/lstm_29/while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split20
.sequential_13/lstm_29/while/lstm_cell_29/splitÛ
0sequential_13/lstm_29/while/lstm_cell_29/SigmoidSigmoid7sequential_13/lstm_29/while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´22
0sequential_13/lstm_29/while/lstm_cell_29/Sigmoidß
2sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_1Sigmoid7sequential_13/lstm_29/while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´24
2sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_1ù
,sequential_13/lstm_29/while/lstm_cell_29/mulMul6sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_1:y:0)sequential_13_lstm_29_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2.
,sequential_13/lstm_29/while/lstm_cell_29/mulÒ
-sequential_13/lstm_29/while/lstm_cell_29/ReluRelu7sequential_13/lstm_29/while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2/
-sequential_13/lstm_29/while/lstm_cell_29/Relu
.sequential_13/lstm_29/while/lstm_cell_29/mul_1Mul4sequential_13/lstm_29/while/lstm_cell_29/Sigmoid:y:0;sequential_13/lstm_29/while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´20
.sequential_13/lstm_29/while/lstm_cell_29/mul_1
.sequential_13/lstm_29/while/lstm_cell_29/add_1AddV20sequential_13/lstm_29/while/lstm_cell_29/mul:z:02sequential_13/lstm_29/while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´20
.sequential_13/lstm_29/while/lstm_cell_29/add_1ß
2sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_2Sigmoid7sequential_13/lstm_29/while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´24
2sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_2Ñ
/sequential_13/lstm_29/while/lstm_cell_29/Relu_1Relu2sequential_13/lstm_29/while/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´21
/sequential_13/lstm_29/while/lstm_cell_29/Relu_1
.sequential_13/lstm_29/while/lstm_cell_29/mul_2Mul6sequential_13/lstm_29/while/lstm_cell_29/Sigmoid_2:y:0=sequential_13/lstm_29/while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´20
.sequential_13/lstm_29/while/lstm_cell_29/mul_2Î
@sequential_13/lstm_29/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_13_lstm_29_while_placeholder_1'sequential_13_lstm_29_while_placeholder2sequential_13/lstm_29/while/lstm_cell_29/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_13/lstm_29/while/TensorArrayV2Write/TensorListSetItem
!sequential_13/lstm_29/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_13/lstm_29/while/add/yÁ
sequential_13/lstm_29/while/addAddV2'sequential_13_lstm_29_while_placeholder*sequential_13/lstm_29/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_29/while/add
#sequential_13/lstm_29/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_13/lstm_29/while/add_1/yä
!sequential_13/lstm_29/while/add_1AddV2Dsequential_13_lstm_29_while_sequential_13_lstm_29_while_loop_counter,sequential_13/lstm_29/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_29/while/add_1Ã
$sequential_13/lstm_29/while/IdentityIdentity%sequential_13/lstm_29/while/add_1:z:0!^sequential_13/lstm_29/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_13/lstm_29/while/Identityì
&sequential_13/lstm_29/while/Identity_1IdentityJsequential_13_lstm_29_while_sequential_13_lstm_29_while_maximum_iterations!^sequential_13/lstm_29/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_29/while/Identity_1Å
&sequential_13/lstm_29/while/Identity_2Identity#sequential_13/lstm_29/while/add:z:0!^sequential_13/lstm_29/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_29/while/Identity_2ò
&sequential_13/lstm_29/while/Identity_3IdentityPsequential_13/lstm_29/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_13/lstm_29/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_29/while/Identity_3æ
&sequential_13/lstm_29/while/Identity_4Identity2sequential_13/lstm_29/while/lstm_cell_29/mul_2:z:0!^sequential_13/lstm_29/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2(
&sequential_13/lstm_29/while/Identity_4æ
&sequential_13/lstm_29/while/Identity_5Identity2sequential_13/lstm_29/while/lstm_cell_29/add_1:z:0!^sequential_13/lstm_29/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2(
&sequential_13/lstm_29/while/Identity_5Ì
 sequential_13/lstm_29/while/NoOpNoOp@^sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp?^sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOpA^sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_13/lstm_29/while/NoOp"U
$sequential_13_lstm_29_while_identity-sequential_13/lstm_29/while/Identity:output:0"Y
&sequential_13_lstm_29_while_identity_1/sequential_13/lstm_29/while/Identity_1:output:0"Y
&sequential_13_lstm_29_while_identity_2/sequential_13/lstm_29/while/Identity_2:output:0"Y
&sequential_13_lstm_29_while_identity_3/sequential_13/lstm_29/while/Identity_3:output:0"Y
&sequential_13_lstm_29_while_identity_4/sequential_13/lstm_29/while/Identity_4:output:0"Y
&sequential_13_lstm_29_while_identity_5/sequential_13/lstm_29/while/Identity_5:output:0"
Hsequential_13_lstm_29_while_lstm_cell_29_biasadd_readvariableop_resourceJsequential_13_lstm_29_while_lstm_cell_29_biasadd_readvariableop_resource_0"
Isequential_13_lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resourceKsequential_13_lstm_29_while_lstm_cell_29_matmul_1_readvariableop_resource_0"
Gsequential_13_lstm_29_while_lstm_cell_29_matmul_readvariableop_resourceIsequential_13_lstm_29_while_lstm_cell_29_matmul_readvariableop_resource_0"
Asequential_13_lstm_29_while_sequential_13_lstm_29_strided_slice_1Csequential_13_lstm_29_while_sequential_13_lstm_29_strided_slice_1_0"
}sequential_13_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_29_tensorarrayunstack_tensorlistfromtensorsequential_13_lstm_29_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_29_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2
?sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp?sequential_13/lstm_29/while/lstm_cell_29/BiasAdd/ReadVariableOp2
>sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp>sequential_13/lstm_29/while/lstm_cell_29/MatMul/ReadVariableOp2
@sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp@sequential_13/lstm_29/while/lstm_cell_29/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1676774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1676774___redundant_placeholder05
1while_while_cond_1676774___redundant_placeholder15
1while_while_cond_1676774___redundant_placeholder25
1while_while_cond_1676774___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
ÃU

D__inference_lstm_31_layer_call_and_return_conditional_losses_1681102

inputs?
+lstm_cell_31_matmul_readvariableop_resource:
´ÐA
-lstm_cell_31_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_31_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_31/BiasAdd/ReadVariableOp¢"lstm_cell_31/MatMul/ReadVariableOp¢$lstm_cell_31/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp­
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul¼
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp©
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul_1 
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/add´
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp­
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim÷
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_31/split
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_1
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_1
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/add_1
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu_1¡
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1681018*
condR
while_cond_1681017*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
È
ù
.__inference_lstm_cell_30_layer_call_fn_1681350

inputs
states_0
states_1
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity

identity_1

identity_2¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_16755812
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
Þ
È
while_cond_1677088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1677088___redundant_placeholder05
1while_while_cond_1677088___redundant_placeholder15
1while_while_cond_1677088___redundant_placeholder25
1while_while_cond_1677088___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
Þ
È
while_cond_1677435
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1677435___redundant_placeholder05
1while_while_cond_1677435___redundant_placeholder15
1while_while_cond_1677435___redundant_placeholder25
1while_while_cond_1677435___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
÷%
î
while_body_1676395
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_31_1676419_0:
´Ð0
while_lstm_cell_31_1676421_0:
´Ð+
while_lstm_cell_31_1676423_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_31_1676419:
´Ð.
while_lstm_cell_31_1676421:
´Ð)
while_lstm_cell_31_1676423:	Ð¢*while/lstm_cell_31/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemè
*while/lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_31_1676419_0while_lstm_cell_31_1676421_0while_lstm_cell_31_1676423_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_16763252,
*while/lstm_cell_31/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_31/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_31/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_31/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_31_1676419while_lstm_cell_31_1676419_0":
while_lstm_cell_31_1676421while_lstm_cell_31_1676421_0":
while_lstm_cell_31_1676423while_lstm_cell_31_1676423_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2X
*while/lstm_cell_31/StatefulPartitionedCall*while/lstm_cell_31/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
Þ
¹
)__inference_lstm_29_layer_call_fn_1679211
inputs_0
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_16750662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
³?
Õ
while_body_1680589
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_31_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_31/BiasAdd/ReadVariableOp¢(while/lstm_cell_31/MatMul/ReadVariableOp¢*while/lstm_cell_31/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp×
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMulÐ
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOpÀ
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMul_1¸
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/addÈ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOpÅ
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/BiasAdd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_31/split
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_1¡
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Reluµ
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_1ª
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/add_1
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_2
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Relu_1¹
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
½U

D__inference_lstm_29_layer_call_and_return_conditional_losses_1676859

inputs>
+lstm_cell_29_matmul_readvariableop_resource:	ÐA
-lstm_cell_29_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_29_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_29/BiasAdd/ReadVariableOp¢"lstm_cell_29/MatMul/ReadVariableOp¢$lstm_cell_29/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp­
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul¼
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp©
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul_1 
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/add´
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp­
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim÷
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_29/split
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_1
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_1
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/add_1
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu_1¡
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1676775*
condR
while_cond_1676774*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷%
î
while_body_1675595
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_30_1675619_0:
´Ð0
while_lstm_cell_30_1675621_0:
´Ð+
while_lstm_cell_30_1675623_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_30_1675619:
´Ð.
while_lstm_cell_30_1675621:
´Ð)
while_lstm_cell_30_1675623:	Ð¢*while/lstm_cell_30/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemè
*while/lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_30_1675619_0while_lstm_cell_30_1675621_0while_lstm_cell_30_1675623_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_16755812,
*while/lstm_cell_30/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_30/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_30/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_30/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_30_1675619while_lstm_cell_30_1675619_0":
while_lstm_cell_30_1675621while_lstm_cell_30_1675621_0":
while_lstm_cell_30_1675623while_lstm_cell_30_1675623_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2X
*while/lstm_cell_30/StatefulPartitionedCall*while/lstm_cell_30/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
ô%
ì
while_body_1675199
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_29_1675223_0:	Ð0
while_lstm_cell_29_1675225_0:
´Ð+
while_lstm_cell_29_1675227_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_29_1675223:	Ð.
while_lstm_cell_29_1675225:
´Ð)
while_lstm_cell_29_1675227:	Ð¢*while/lstm_cell_29/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemè
*while/lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_29_1675223_0while_lstm_cell_29_1675225_0while_lstm_cell_29_1675227_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_16751292,
*while/lstm_cell_29/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_29/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_29/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_29/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_29_1675223while_lstm_cell_29_1675223_0":
while_lstm_cell_29_1675225while_lstm_cell_29_1675225_0":
while_lstm_cell_29_1675227while_lstm_cell_29_1675227_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2X
*while/lstm_cell_29/StatefulPartitionedCall*while/lstm_cell_29/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1679731
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1679731___redundant_placeholder05
1while_while_cond_1679731___redundant_placeholder15
1while_while_cond_1679731___redundant_placeholder25
1while_while_cond_1679731___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
ô

(sequential_13_lstm_29_while_cond_1674496H
Dsequential_13_lstm_29_while_sequential_13_lstm_29_while_loop_counterN
Jsequential_13_lstm_29_while_sequential_13_lstm_29_while_maximum_iterations+
'sequential_13_lstm_29_while_placeholder-
)sequential_13_lstm_29_while_placeholder_1-
)sequential_13_lstm_29_while_placeholder_2-
)sequential_13_lstm_29_while_placeholder_3J
Fsequential_13_lstm_29_while_less_sequential_13_lstm_29_strided_slice_1a
]sequential_13_lstm_29_while_sequential_13_lstm_29_while_cond_1674496___redundant_placeholder0a
]sequential_13_lstm_29_while_sequential_13_lstm_29_while_cond_1674496___redundant_placeholder1a
]sequential_13_lstm_29_while_sequential_13_lstm_29_while_cond_1674496___redundant_placeholder2a
]sequential_13_lstm_29_while_sequential_13_lstm_29_while_cond_1674496___redundant_placeholder3(
$sequential_13_lstm_29_while_identity
Þ
 sequential_13/lstm_29/while/LessLess'sequential_13_lstm_29_while_placeholderFsequential_13_lstm_29_while_less_sequential_13_lstm_29_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_13/lstm_29/while/Less
$sequential_13/lstm_29/while/IdentityIdentity$sequential_13/lstm_29/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_13/lstm_29/while/Identity"U
$sequential_13_lstm_29_while_identity-sequential_13/lstm_29/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
V
 
D__inference_lstm_30_layer_call_and_return_conditional_losses_1680173
inputs_0?
+lstm_cell_30_matmul_readvariableop_resource:
´ÐA
-lstm_cell_30_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_30_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_30/BiasAdd/ReadVariableOp¢"lstm_cell_30/MatMul/ReadVariableOp¢$lstm_cell_30/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp­
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul¼
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp©
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul_1 
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/add´
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp­
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim÷
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_30/split
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_1
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_1
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/add_1
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu_1¡
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1680089*
condR
while_cond_1680088*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
inputs/0
èJ
Õ

lstm_30_while_body_1678900,
(lstm_30_while_lstm_30_while_loop_counter2
.lstm_30_while_lstm_30_while_maximum_iterations
lstm_30_while_placeholder
lstm_30_while_placeholder_1
lstm_30_while_placeholder_2
lstm_30_while_placeholder_3+
'lstm_30_while_lstm_30_strided_slice_1_0g
clstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0:
´ÐQ
=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0:
´ÐK
<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0:	Ð
lstm_30_while_identity
lstm_30_while_identity_1
lstm_30_while_identity_2
lstm_30_while_identity_3
lstm_30_while_identity_4
lstm_30_while_identity_5)
%lstm_30_while_lstm_30_strided_slice_1e
alstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensorM
9lstm_30_while_lstm_cell_30_matmul_readvariableop_resource:
´ÐO
;lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐI
:lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource:	Ð¢1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp¢0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp¢2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpÓ
?lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2A
?lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_30/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0lstm_30_while_placeholderHlstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype023
1lstm_30/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype022
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp÷
!lstm_30/while/lstm_cell_30/MatMulMatMul8lstm_30/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2#
!lstm_30/while/lstm_cell_30/MatMulè
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype024
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpà
#lstm_30/while/lstm_cell_30/MatMul_1MatMullstm_30_while_placeholder_2:lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2%
#lstm_30/while/lstm_cell_30/MatMul_1Ø
lstm_30/while/lstm_cell_30/addAddV2+lstm_30/while/lstm_cell_30/MatMul:product:0-lstm_30/while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2 
lstm_30/while/lstm_cell_30/addà
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype023
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOpå
"lstm_30/while/lstm_cell_30/BiasAddBiasAdd"lstm_30/while/lstm_cell_30/add:z:09lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2$
"lstm_30/while/lstm_cell_30/BiasAdd
*lstm_30/while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_30/while/lstm_cell_30/split/split_dim¯
 lstm_30/while/lstm_cell_30/splitSplit3lstm_30/while/lstm_cell_30/split/split_dim:output:0+lstm_30/while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2"
 lstm_30/while/lstm_cell_30/split±
"lstm_30/while/lstm_cell_30/SigmoidSigmoid)lstm_30/while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2$
"lstm_30/while/lstm_cell_30/Sigmoidµ
$lstm_30/while/lstm_cell_30/Sigmoid_1Sigmoid)lstm_30/while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_30/while/lstm_cell_30/Sigmoid_1Á
lstm_30/while/lstm_cell_30/mulMul(lstm_30/while/lstm_cell_30/Sigmoid_1:y:0lstm_30_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_30/while/lstm_cell_30/mul¨
lstm_30/while/lstm_cell_30/ReluRelu)lstm_30/while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
lstm_30/while/lstm_cell_30/ReluÕ
 lstm_30/while/lstm_cell_30/mul_1Mul&lstm_30/while/lstm_cell_30/Sigmoid:y:0-lstm_30/while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_30/while/lstm_cell_30/mul_1Ê
 lstm_30/while/lstm_cell_30/add_1AddV2"lstm_30/while/lstm_cell_30/mul:z:0$lstm_30/while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_30/while/lstm_cell_30/add_1µ
$lstm_30/while/lstm_cell_30/Sigmoid_2Sigmoid)lstm_30/while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_30/while/lstm_cell_30/Sigmoid_2§
!lstm_30/while/lstm_cell_30/Relu_1Relu$lstm_30/while/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!lstm_30/while/lstm_cell_30/Relu_1Ù
 lstm_30/while/lstm_cell_30/mul_2Mul(lstm_30/while/lstm_cell_30/Sigmoid_2:y:0/lstm_30/while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_30/while/lstm_cell_30/mul_2
2lstm_30/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_30_while_placeholder_1lstm_30_while_placeholder$lstm_30/while/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_30/while/TensorArrayV2Write/TensorListSetIteml
lstm_30/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_30/while/add/y
lstm_30/while/addAddV2lstm_30_while_placeholderlstm_30/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_30/while/addp
lstm_30/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_30/while/add_1/y
lstm_30/while/add_1AddV2(lstm_30_while_lstm_30_while_loop_counterlstm_30/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_30/while/add_1
lstm_30/while/IdentityIdentitylstm_30/while/add_1:z:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity¦
lstm_30/while/Identity_1Identity.lstm_30_while_lstm_30_while_maximum_iterations^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_1
lstm_30/while/Identity_2Identitylstm_30/while/add:z:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_2º
lstm_30/while/Identity_3IdentityBlstm_30/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_3®
lstm_30/while/Identity_4Identity$lstm_30/while/lstm_cell_30/mul_2:z:0^lstm_30/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/while/Identity_4®
lstm_30/while/Identity_5Identity$lstm_30/while/lstm_cell_30/add_1:z:0^lstm_30/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/while/Identity_5
lstm_30/while/NoOpNoOp2^lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp1^lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp3^lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_30/while/NoOp"9
lstm_30_while_identitylstm_30/while/Identity:output:0"=
lstm_30_while_identity_1!lstm_30/while/Identity_1:output:0"=
lstm_30_while_identity_2!lstm_30/while/Identity_2:output:0"=
lstm_30_while_identity_3!lstm_30/while/Identity_3:output:0"=
lstm_30_while_identity_4!lstm_30/while/Identity_4:output:0"=
lstm_30_while_identity_5!lstm_30/while/Identity_5:output:0"P
%lstm_30_while_lstm_30_strided_slice_1'lstm_30_while_lstm_30_strided_slice_1_0"z
:lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0"|
;lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0"x
9lstm_30_while_lstm_cell_30_matmul_readvariableop_resource;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0"È
alstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensorclstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2f
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp2d
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp2h
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
·
¸
)__inference_lstm_30_layer_call_fn_1679876

inputs
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_16770162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
ú?

D__inference_lstm_29_layer_call_and_return_conditional_losses_1675268

inputs'
lstm_cell_29_1675186:	Ð(
lstm_cell_29_1675188:
´Ð#
lstm_cell_29_1675190:	Ð
identity¢$lstm_cell_29/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¤
$lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_29_1675186lstm_cell_29_1675188lstm_cell_29_1675190*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_16751292&
$lstm_cell_29/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÉ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_29_1675186lstm_cell_29_1675188lstm_cell_29_1675190*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1675199*
condR
while_cond_1675198*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identity}
NoOpNoOp%^lstm_cell_29/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_29/StatefulPartitionedCall$lstm_cell_29/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç^

(sequential_13_lstm_30_while_body_1674637H
Dsequential_13_lstm_30_while_sequential_13_lstm_30_while_loop_counterN
Jsequential_13_lstm_30_while_sequential_13_lstm_30_while_maximum_iterations+
'sequential_13_lstm_30_while_placeholder-
)sequential_13_lstm_30_while_placeholder_1-
)sequential_13_lstm_30_while_placeholder_2-
)sequential_13_lstm_30_while_placeholder_3G
Csequential_13_lstm_30_while_sequential_13_lstm_30_strided_slice_1_0
sequential_13_lstm_30_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_30_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_13_lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0:
´Ð_
Ksequential_13_lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0:
´ÐY
Jsequential_13_lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0:	Ð(
$sequential_13_lstm_30_while_identity*
&sequential_13_lstm_30_while_identity_1*
&sequential_13_lstm_30_while_identity_2*
&sequential_13_lstm_30_while_identity_3*
&sequential_13_lstm_30_while_identity_4*
&sequential_13_lstm_30_while_identity_5E
Asequential_13_lstm_30_while_sequential_13_lstm_30_strided_slice_1
}sequential_13_lstm_30_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_30_tensorarrayunstack_tensorlistfromtensor[
Gsequential_13_lstm_30_while_lstm_cell_30_matmul_readvariableop_resource:
´Ð]
Isequential_13_lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐW
Hsequential_13_lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource:	Ð¢?sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp¢>sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp¢@sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpï
Msequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2O
Msequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shapeØ
?sequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_13_lstm_30_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_30_tensorarrayunstack_tensorlistfromtensor_0'sequential_13_lstm_30_while_placeholderVsequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02A
?sequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItem
>sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOpIsequential_13_lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02@
>sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp¯
/sequential_13/lstm_30/while/lstm_cell_30/MatMulMatMulFsequential_13/lstm_30/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ21
/sequential_13/lstm_30/while/lstm_cell_30/MatMul
@sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOpKsequential_13_lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02B
@sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp
1sequential_13/lstm_30/while/lstm_cell_30/MatMul_1MatMul)sequential_13_lstm_30_while_placeholder_2Hsequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ23
1sequential_13/lstm_30/while/lstm_cell_30/MatMul_1
,sequential_13/lstm_30/while/lstm_cell_30/addAddV29sequential_13/lstm_30/while/lstm_cell_30/MatMul:product:0;sequential_13/lstm_30/while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2.
,sequential_13/lstm_30/while/lstm_cell_30/add
?sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOpJsequential_13_lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02A
?sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp
0sequential_13/lstm_30/while/lstm_cell_30/BiasAddBiasAdd0sequential_13/lstm_30/while/lstm_cell_30/add:z:0Gsequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ22
0sequential_13/lstm_30/while/lstm_cell_30/BiasAdd¶
8sequential_13/lstm_30/while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_13/lstm_30/while/lstm_cell_30/split/split_dimç
.sequential_13/lstm_30/while/lstm_cell_30/splitSplitAsequential_13/lstm_30/while/lstm_cell_30/split/split_dim:output:09sequential_13/lstm_30/while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split20
.sequential_13/lstm_30/while/lstm_cell_30/splitÛ
0sequential_13/lstm_30/while/lstm_cell_30/SigmoidSigmoid7sequential_13/lstm_30/while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´22
0sequential_13/lstm_30/while/lstm_cell_30/Sigmoidß
2sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_1Sigmoid7sequential_13/lstm_30/while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´24
2sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_1ù
,sequential_13/lstm_30/while/lstm_cell_30/mulMul6sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_1:y:0)sequential_13_lstm_30_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2.
,sequential_13/lstm_30/while/lstm_cell_30/mulÒ
-sequential_13/lstm_30/while/lstm_cell_30/ReluRelu7sequential_13/lstm_30/while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2/
-sequential_13/lstm_30/while/lstm_cell_30/Relu
.sequential_13/lstm_30/while/lstm_cell_30/mul_1Mul4sequential_13/lstm_30/while/lstm_cell_30/Sigmoid:y:0;sequential_13/lstm_30/while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´20
.sequential_13/lstm_30/while/lstm_cell_30/mul_1
.sequential_13/lstm_30/while/lstm_cell_30/add_1AddV20sequential_13/lstm_30/while/lstm_cell_30/mul:z:02sequential_13/lstm_30/while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´20
.sequential_13/lstm_30/while/lstm_cell_30/add_1ß
2sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_2Sigmoid7sequential_13/lstm_30/while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´24
2sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_2Ñ
/sequential_13/lstm_30/while/lstm_cell_30/Relu_1Relu2sequential_13/lstm_30/while/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´21
/sequential_13/lstm_30/while/lstm_cell_30/Relu_1
.sequential_13/lstm_30/while/lstm_cell_30/mul_2Mul6sequential_13/lstm_30/while/lstm_cell_30/Sigmoid_2:y:0=sequential_13/lstm_30/while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´20
.sequential_13/lstm_30/while/lstm_cell_30/mul_2Î
@sequential_13/lstm_30/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_13_lstm_30_while_placeholder_1'sequential_13_lstm_30_while_placeholder2sequential_13/lstm_30/while/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_13/lstm_30/while/TensorArrayV2Write/TensorListSetItem
!sequential_13/lstm_30/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_13/lstm_30/while/add/yÁ
sequential_13/lstm_30/while/addAddV2'sequential_13_lstm_30_while_placeholder*sequential_13/lstm_30/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_30/while/add
#sequential_13/lstm_30/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_13/lstm_30/while/add_1/yä
!sequential_13/lstm_30/while/add_1AddV2Dsequential_13_lstm_30_while_sequential_13_lstm_30_while_loop_counter,sequential_13/lstm_30/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_30/while/add_1Ã
$sequential_13/lstm_30/while/IdentityIdentity%sequential_13/lstm_30/while/add_1:z:0!^sequential_13/lstm_30/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_13/lstm_30/while/Identityì
&sequential_13/lstm_30/while/Identity_1IdentityJsequential_13_lstm_30_while_sequential_13_lstm_30_while_maximum_iterations!^sequential_13/lstm_30/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_30/while/Identity_1Å
&sequential_13/lstm_30/while/Identity_2Identity#sequential_13/lstm_30/while/add:z:0!^sequential_13/lstm_30/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_30/while/Identity_2ò
&sequential_13/lstm_30/while/Identity_3IdentityPsequential_13/lstm_30/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_13/lstm_30/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_30/while/Identity_3æ
&sequential_13/lstm_30/while/Identity_4Identity2sequential_13/lstm_30/while/lstm_cell_30/mul_2:z:0!^sequential_13/lstm_30/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2(
&sequential_13/lstm_30/while/Identity_4æ
&sequential_13/lstm_30/while/Identity_5Identity2sequential_13/lstm_30/while/lstm_cell_30/add_1:z:0!^sequential_13/lstm_30/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2(
&sequential_13/lstm_30/while/Identity_5Ì
 sequential_13/lstm_30/while/NoOpNoOp@^sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp?^sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOpA^sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_13/lstm_30/while/NoOp"U
$sequential_13_lstm_30_while_identity-sequential_13/lstm_30/while/Identity:output:0"Y
&sequential_13_lstm_30_while_identity_1/sequential_13/lstm_30/while/Identity_1:output:0"Y
&sequential_13_lstm_30_while_identity_2/sequential_13/lstm_30/while/Identity_2:output:0"Y
&sequential_13_lstm_30_while_identity_3/sequential_13/lstm_30/while/Identity_3:output:0"Y
&sequential_13_lstm_30_while_identity_4/sequential_13/lstm_30/while/Identity_4:output:0"Y
&sequential_13_lstm_30_while_identity_5/sequential_13/lstm_30/while/Identity_5:output:0"
Hsequential_13_lstm_30_while_lstm_cell_30_biasadd_readvariableop_resourceJsequential_13_lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0"
Isequential_13_lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resourceKsequential_13_lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0"
Gsequential_13_lstm_30_while_lstm_cell_30_matmul_readvariableop_resourceIsequential_13_lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0"
Asequential_13_lstm_30_while_sequential_13_lstm_30_strided_slice_1Csequential_13_lstm_30_while_sequential_13_lstm_30_strided_slice_1_0"
}sequential_13_lstm_30_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_30_tensorarrayunstack_tensorlistfromtensorsequential_13_lstm_30_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_30_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2
?sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp?sequential_13/lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp2
>sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp>sequential_13/lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp2
@sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp@sequential_13/lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
üU

D__inference_lstm_29_layer_call_and_return_conditional_losses_1679387
inputs_0>
+lstm_cell_29_matmul_readvariableop_resource:	ÐA
-lstm_cell_29_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_29_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_29/BiasAdd/ReadVariableOp¢"lstm_cell_29/MatMul/ReadVariableOp¢$lstm_cell_29/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp­
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul¼
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp©
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul_1 
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/add´
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp­
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim÷
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_29/split
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_1
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_1
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/add_1
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu_1¡
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1679303*
condR
while_cond_1679302*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0


*__inference_dense_32_layer_call_fn_1681205

inputs
unknown:	´
	unknown_0:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_16772622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Þ
È
while_cond_1675198
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1675198___redundant_placeholder05
1while_while_cond_1675198___redundant_placeholder15
1while_while_cond_1675198___redundant_placeholder25
1while_while_cond_1675198___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
±Ë
²
"__inference__wrapped_model_1674916
lstm_29_inputT
Asequential_13_lstm_29_lstm_cell_29_matmul_readvariableop_resource:	ÐW
Csequential_13_lstm_29_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐQ
Bsequential_13_lstm_29_lstm_cell_29_biasadd_readvariableop_resource:	ÐU
Asequential_13_lstm_30_lstm_cell_30_matmul_readvariableop_resource:
´ÐW
Csequential_13_lstm_30_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐQ
Bsequential_13_lstm_30_lstm_cell_30_biasadd_readvariableop_resource:	ÐU
Asequential_13_lstm_31_lstm_cell_31_matmul_readvariableop_resource:
´ÐW
Csequential_13_lstm_31_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐQ
Bsequential_13_lstm_31_lstm_cell_31_biasadd_readvariableop_resource:	ÐL
8sequential_13_dense_31_tensordot_readvariableop_resource:
´´E
6sequential_13_dense_31_biasadd_readvariableop_resource:	´K
8sequential_13_dense_32_tensordot_readvariableop_resource:	´D
6sequential_13_dense_32_biasadd_readvariableop_resource:
identity¢-sequential_13/dense_31/BiasAdd/ReadVariableOp¢/sequential_13/dense_31/Tensordot/ReadVariableOp¢-sequential_13/dense_32/BiasAdd/ReadVariableOp¢/sequential_13/dense_32/Tensordot/ReadVariableOp¢9sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp¢8sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp¢:sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp¢sequential_13/lstm_29/while¢9sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp¢8sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp¢:sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp¢sequential_13/lstm_30/while¢9sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp¢8sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp¢:sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp¢sequential_13/lstm_31/whilew
sequential_13/lstm_29/ShapeShapelstm_29_input*
T0*
_output_shapes
:2
sequential_13/lstm_29/Shape 
)sequential_13/lstm_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_13/lstm_29/strided_slice/stack¤
+sequential_13/lstm_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_29/strided_slice/stack_1¤
+sequential_13/lstm_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_29/strided_slice/stack_2æ
#sequential_13/lstm_29/strided_sliceStridedSlice$sequential_13/lstm_29/Shape:output:02sequential_13/lstm_29/strided_slice/stack:output:04sequential_13/lstm_29/strided_slice/stack_1:output:04sequential_13/lstm_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_13/lstm_29/strided_slice
$sequential_13/lstm_29/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2&
$sequential_13/lstm_29/zeros/packed/1Û
"sequential_13/lstm_29/zeros/packedPack,sequential_13/lstm_29/strided_slice:output:0-sequential_13/lstm_29/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_13/lstm_29/zeros/packed
!sequential_13/lstm_29/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_13/lstm_29/zeros/ConstÎ
sequential_13/lstm_29/zerosFill+sequential_13/lstm_29/zeros/packed:output:0*sequential_13/lstm_29/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
sequential_13/lstm_29/zeros
&sequential_13/lstm_29/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2(
&sequential_13/lstm_29/zeros_1/packed/1á
$sequential_13/lstm_29/zeros_1/packedPack,sequential_13/lstm_29/strided_slice:output:0/sequential_13/lstm_29/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_13/lstm_29/zeros_1/packed
#sequential_13/lstm_29/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_13/lstm_29/zeros_1/ConstÖ
sequential_13/lstm_29/zeros_1Fill-sequential_13/lstm_29/zeros_1/packed:output:0,sequential_13/lstm_29/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
sequential_13/lstm_29/zeros_1¡
$sequential_13/lstm_29/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_13/lstm_29/transpose/permÃ
sequential_13/lstm_29/transpose	Transposelstm_29_input-sequential_13/lstm_29/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_13/lstm_29/transpose
sequential_13/lstm_29/Shape_1Shape#sequential_13/lstm_29/transpose:y:0*
T0*
_output_shapes
:2
sequential_13/lstm_29/Shape_1¤
+sequential_13/lstm_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_29/strided_slice_1/stack¨
-sequential_13/lstm_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_29/strided_slice_1/stack_1¨
-sequential_13/lstm_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_29/strided_slice_1/stack_2ò
%sequential_13/lstm_29/strided_slice_1StridedSlice&sequential_13/lstm_29/Shape_1:output:04sequential_13/lstm_29/strided_slice_1/stack:output:06sequential_13/lstm_29/strided_slice_1/stack_1:output:06sequential_13/lstm_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_13/lstm_29/strided_slice_1±
1sequential_13/lstm_29/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_13/lstm_29/TensorArrayV2/element_shape
#sequential_13/lstm_29/TensorArrayV2TensorListReserve:sequential_13/lstm_29/TensorArrayV2/element_shape:output:0.sequential_13/lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_13/lstm_29/TensorArrayV2ë
Ksequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_13/lstm_29/transpose:y:0Tsequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensor¤
+sequential_13/lstm_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_29/strided_slice_2/stack¨
-sequential_13/lstm_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_29/strided_slice_2/stack_1¨
-sequential_13/lstm_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_29/strided_slice_2/stack_2
%sequential_13/lstm_29/strided_slice_2StridedSlice#sequential_13/lstm_29/transpose:y:04sequential_13/lstm_29/strided_slice_2/stack:output:06sequential_13/lstm_29/strided_slice_2/stack_1:output:06sequential_13/lstm_29/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_13/lstm_29/strided_slice_2÷
8sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOpReadVariableOpAsequential_13_lstm_29_lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02:
8sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp
)sequential_13/lstm_29/lstm_cell_29/MatMulMatMul.sequential_13/lstm_29/strided_slice_2:output:0@sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2+
)sequential_13/lstm_29/lstm_cell_29/MatMulþ
:sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOpCsequential_13_lstm_29_lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02<
:sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp
+sequential_13/lstm_29/lstm_cell_29/MatMul_1MatMul$sequential_13/lstm_29/zeros:output:0Bsequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2-
+sequential_13/lstm_29/lstm_cell_29/MatMul_1ø
&sequential_13/lstm_29/lstm_cell_29/addAddV23sequential_13/lstm_29/lstm_cell_29/MatMul:product:05sequential_13/lstm_29/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2(
&sequential_13/lstm_29/lstm_cell_29/addö
9sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOpBsequential_13_lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02;
9sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp
*sequential_13/lstm_29/lstm_cell_29/BiasAddBiasAdd*sequential_13/lstm_29/lstm_cell_29/add:z:0Asequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2,
*sequential_13/lstm_29/lstm_cell_29/BiasAddª
2sequential_13/lstm_29/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_13/lstm_29/lstm_cell_29/split/split_dimÏ
(sequential_13/lstm_29/lstm_cell_29/splitSplit;sequential_13/lstm_29/lstm_cell_29/split/split_dim:output:03sequential_13/lstm_29/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2*
(sequential_13/lstm_29/lstm_cell_29/splitÉ
*sequential_13/lstm_29/lstm_cell_29/SigmoidSigmoid1sequential_13/lstm_29/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2,
*sequential_13/lstm_29/lstm_cell_29/SigmoidÍ
,sequential_13/lstm_29/lstm_cell_29/Sigmoid_1Sigmoid1sequential_13/lstm_29/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2.
,sequential_13/lstm_29/lstm_cell_29/Sigmoid_1ä
&sequential_13/lstm_29/lstm_cell_29/mulMul0sequential_13/lstm_29/lstm_cell_29/Sigmoid_1:y:0&sequential_13/lstm_29/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2(
&sequential_13/lstm_29/lstm_cell_29/mulÀ
'sequential_13/lstm_29/lstm_cell_29/ReluRelu1sequential_13/lstm_29/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2)
'sequential_13/lstm_29/lstm_cell_29/Reluõ
(sequential_13/lstm_29/lstm_cell_29/mul_1Mul.sequential_13/lstm_29/lstm_cell_29/Sigmoid:y:05sequential_13/lstm_29/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2*
(sequential_13/lstm_29/lstm_cell_29/mul_1ê
(sequential_13/lstm_29/lstm_cell_29/add_1AddV2*sequential_13/lstm_29/lstm_cell_29/mul:z:0,sequential_13/lstm_29/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2*
(sequential_13/lstm_29/lstm_cell_29/add_1Í
,sequential_13/lstm_29/lstm_cell_29/Sigmoid_2Sigmoid1sequential_13/lstm_29/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2.
,sequential_13/lstm_29/lstm_cell_29/Sigmoid_2¿
)sequential_13/lstm_29/lstm_cell_29/Relu_1Relu,sequential_13/lstm_29/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2+
)sequential_13/lstm_29/lstm_cell_29/Relu_1ù
(sequential_13/lstm_29/lstm_cell_29/mul_2Mul0sequential_13/lstm_29/lstm_cell_29/Sigmoid_2:y:07sequential_13/lstm_29/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2*
(sequential_13/lstm_29/lstm_cell_29/mul_2»
3sequential_13/lstm_29/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   25
3sequential_13/lstm_29/TensorArrayV2_1/element_shape
%sequential_13/lstm_29/TensorArrayV2_1TensorListReserve<sequential_13/lstm_29/TensorArrayV2_1/element_shape:output:0.sequential_13/lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_13/lstm_29/TensorArrayV2_1z
sequential_13/lstm_29/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_13/lstm_29/time«
.sequential_13/lstm_29/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_13/lstm_29/while/maximum_iterations
(sequential_13/lstm_29/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_13/lstm_29/while/loop_counterÛ
sequential_13/lstm_29/whileWhile1sequential_13/lstm_29/while/loop_counter:output:07sequential_13/lstm_29/while/maximum_iterations:output:0#sequential_13/lstm_29/time:output:0.sequential_13/lstm_29/TensorArrayV2_1:handle:0$sequential_13/lstm_29/zeros:output:0&sequential_13/lstm_29/zeros_1:output:0.sequential_13/lstm_29/strided_slice_1:output:0Msequential_13/lstm_29/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_13_lstm_29_lstm_cell_29_matmul_readvariableop_resourceCsequential_13_lstm_29_lstm_cell_29_matmul_1_readvariableop_resourceBsequential_13_lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_13_lstm_29_while_body_1674497*4
cond,R*
(sequential_13_lstm_29_while_cond_1674496*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
sequential_13/lstm_29/whileá
Fsequential_13/lstm_29/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2H
Fsequential_13/lstm_29/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_13/lstm_29/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_13/lstm_29/while:output:3Osequential_13/lstm_29/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02:
8sequential_13/lstm_29/TensorArrayV2Stack/TensorListStack­
+sequential_13/lstm_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_13/lstm_29/strided_slice_3/stack¨
-sequential_13/lstm_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_13/lstm_29/strided_slice_3/stack_1¨
-sequential_13/lstm_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_29/strided_slice_3/stack_2
%sequential_13/lstm_29/strided_slice_3StridedSliceAsequential_13/lstm_29/TensorArrayV2Stack/TensorListStack:tensor:04sequential_13/lstm_29/strided_slice_3/stack:output:06sequential_13/lstm_29/strided_slice_3/stack_1:output:06sequential_13/lstm_29/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2'
%sequential_13/lstm_29/strided_slice_3¥
&sequential_13/lstm_29/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_13/lstm_29/transpose_1/permþ
!sequential_13/lstm_29/transpose_1	TransposeAsequential_13/lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_13/lstm_29/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!sequential_13/lstm_29/transpose_1
sequential_13/lstm_29/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_13/lstm_29/runtime°
!sequential_13/dropout_47/IdentityIdentity%sequential_13/lstm_29/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!sequential_13/dropout_47/Identity
sequential_13/lstm_30/ShapeShape*sequential_13/dropout_47/Identity:output:0*
T0*
_output_shapes
:2
sequential_13/lstm_30/Shape 
)sequential_13/lstm_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_13/lstm_30/strided_slice/stack¤
+sequential_13/lstm_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_30/strided_slice/stack_1¤
+sequential_13/lstm_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_30/strided_slice/stack_2æ
#sequential_13/lstm_30/strided_sliceStridedSlice$sequential_13/lstm_30/Shape:output:02sequential_13/lstm_30/strided_slice/stack:output:04sequential_13/lstm_30/strided_slice/stack_1:output:04sequential_13/lstm_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_13/lstm_30/strided_slice
$sequential_13/lstm_30/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2&
$sequential_13/lstm_30/zeros/packed/1Û
"sequential_13/lstm_30/zeros/packedPack,sequential_13/lstm_30/strided_slice:output:0-sequential_13/lstm_30/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_13/lstm_30/zeros/packed
!sequential_13/lstm_30/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_13/lstm_30/zeros/ConstÎ
sequential_13/lstm_30/zerosFill+sequential_13/lstm_30/zeros/packed:output:0*sequential_13/lstm_30/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
sequential_13/lstm_30/zeros
&sequential_13/lstm_30/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2(
&sequential_13/lstm_30/zeros_1/packed/1á
$sequential_13/lstm_30/zeros_1/packedPack,sequential_13/lstm_30/strided_slice:output:0/sequential_13/lstm_30/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_13/lstm_30/zeros_1/packed
#sequential_13/lstm_30/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_13/lstm_30/zeros_1/ConstÖ
sequential_13/lstm_30/zeros_1Fill-sequential_13/lstm_30/zeros_1/packed:output:0,sequential_13/lstm_30/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
sequential_13/lstm_30/zeros_1¡
$sequential_13/lstm_30/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_13/lstm_30/transpose/permá
sequential_13/lstm_30/transpose	Transpose*sequential_13/dropout_47/Identity:output:0-sequential_13/lstm_30/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
sequential_13/lstm_30/transpose
sequential_13/lstm_30/Shape_1Shape#sequential_13/lstm_30/transpose:y:0*
T0*
_output_shapes
:2
sequential_13/lstm_30/Shape_1¤
+sequential_13/lstm_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_30/strided_slice_1/stack¨
-sequential_13/lstm_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_30/strided_slice_1/stack_1¨
-sequential_13/lstm_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_30/strided_slice_1/stack_2ò
%sequential_13/lstm_30/strided_slice_1StridedSlice&sequential_13/lstm_30/Shape_1:output:04sequential_13/lstm_30/strided_slice_1/stack:output:06sequential_13/lstm_30/strided_slice_1/stack_1:output:06sequential_13/lstm_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_13/lstm_30/strided_slice_1±
1sequential_13/lstm_30/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_13/lstm_30/TensorArrayV2/element_shape
#sequential_13/lstm_30/TensorArrayV2TensorListReserve:sequential_13/lstm_30/TensorArrayV2/element_shape:output:0.sequential_13/lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_13/lstm_30/TensorArrayV2ë
Ksequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2M
Ksequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_13/lstm_30/transpose:y:0Tsequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensor¤
+sequential_13/lstm_30/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_30/strided_slice_2/stack¨
-sequential_13/lstm_30/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_30/strided_slice_2/stack_1¨
-sequential_13/lstm_30/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_30/strided_slice_2/stack_2
%sequential_13/lstm_30/strided_slice_2StridedSlice#sequential_13/lstm_30/transpose:y:04sequential_13/lstm_30/strided_slice_2/stack:output:06sequential_13/lstm_30/strided_slice_2/stack_1:output:06sequential_13/lstm_30/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2'
%sequential_13/lstm_30/strided_slice_2ø
8sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOpReadVariableOpAsequential_13_lstm_30_lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02:
8sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp
)sequential_13/lstm_30/lstm_cell_30/MatMulMatMul.sequential_13/lstm_30/strided_slice_2:output:0@sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2+
)sequential_13/lstm_30/lstm_cell_30/MatMulþ
:sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOpCsequential_13_lstm_30_lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02<
:sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp
+sequential_13/lstm_30/lstm_cell_30/MatMul_1MatMul$sequential_13/lstm_30/zeros:output:0Bsequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2-
+sequential_13/lstm_30/lstm_cell_30/MatMul_1ø
&sequential_13/lstm_30/lstm_cell_30/addAddV23sequential_13/lstm_30/lstm_cell_30/MatMul:product:05sequential_13/lstm_30/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2(
&sequential_13/lstm_30/lstm_cell_30/addö
9sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOpBsequential_13_lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02;
9sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp
*sequential_13/lstm_30/lstm_cell_30/BiasAddBiasAdd*sequential_13/lstm_30/lstm_cell_30/add:z:0Asequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2,
*sequential_13/lstm_30/lstm_cell_30/BiasAddª
2sequential_13/lstm_30/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_13/lstm_30/lstm_cell_30/split/split_dimÏ
(sequential_13/lstm_30/lstm_cell_30/splitSplit;sequential_13/lstm_30/lstm_cell_30/split/split_dim:output:03sequential_13/lstm_30/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2*
(sequential_13/lstm_30/lstm_cell_30/splitÉ
*sequential_13/lstm_30/lstm_cell_30/SigmoidSigmoid1sequential_13/lstm_30/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2,
*sequential_13/lstm_30/lstm_cell_30/SigmoidÍ
,sequential_13/lstm_30/lstm_cell_30/Sigmoid_1Sigmoid1sequential_13/lstm_30/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2.
,sequential_13/lstm_30/lstm_cell_30/Sigmoid_1ä
&sequential_13/lstm_30/lstm_cell_30/mulMul0sequential_13/lstm_30/lstm_cell_30/Sigmoid_1:y:0&sequential_13/lstm_30/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2(
&sequential_13/lstm_30/lstm_cell_30/mulÀ
'sequential_13/lstm_30/lstm_cell_30/ReluRelu1sequential_13/lstm_30/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2)
'sequential_13/lstm_30/lstm_cell_30/Reluõ
(sequential_13/lstm_30/lstm_cell_30/mul_1Mul.sequential_13/lstm_30/lstm_cell_30/Sigmoid:y:05sequential_13/lstm_30/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2*
(sequential_13/lstm_30/lstm_cell_30/mul_1ê
(sequential_13/lstm_30/lstm_cell_30/add_1AddV2*sequential_13/lstm_30/lstm_cell_30/mul:z:0,sequential_13/lstm_30/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2*
(sequential_13/lstm_30/lstm_cell_30/add_1Í
,sequential_13/lstm_30/lstm_cell_30/Sigmoid_2Sigmoid1sequential_13/lstm_30/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2.
,sequential_13/lstm_30/lstm_cell_30/Sigmoid_2¿
)sequential_13/lstm_30/lstm_cell_30/Relu_1Relu,sequential_13/lstm_30/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2+
)sequential_13/lstm_30/lstm_cell_30/Relu_1ù
(sequential_13/lstm_30/lstm_cell_30/mul_2Mul0sequential_13/lstm_30/lstm_cell_30/Sigmoid_2:y:07sequential_13/lstm_30/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2*
(sequential_13/lstm_30/lstm_cell_30/mul_2»
3sequential_13/lstm_30/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   25
3sequential_13/lstm_30/TensorArrayV2_1/element_shape
%sequential_13/lstm_30/TensorArrayV2_1TensorListReserve<sequential_13/lstm_30/TensorArrayV2_1/element_shape:output:0.sequential_13/lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_13/lstm_30/TensorArrayV2_1z
sequential_13/lstm_30/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_13/lstm_30/time«
.sequential_13/lstm_30/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_13/lstm_30/while/maximum_iterations
(sequential_13/lstm_30/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_13/lstm_30/while/loop_counterÛ
sequential_13/lstm_30/whileWhile1sequential_13/lstm_30/while/loop_counter:output:07sequential_13/lstm_30/while/maximum_iterations:output:0#sequential_13/lstm_30/time:output:0.sequential_13/lstm_30/TensorArrayV2_1:handle:0$sequential_13/lstm_30/zeros:output:0&sequential_13/lstm_30/zeros_1:output:0.sequential_13/lstm_30/strided_slice_1:output:0Msequential_13/lstm_30/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_13_lstm_30_lstm_cell_30_matmul_readvariableop_resourceCsequential_13_lstm_30_lstm_cell_30_matmul_1_readvariableop_resourceBsequential_13_lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_13_lstm_30_while_body_1674637*4
cond,R*
(sequential_13_lstm_30_while_cond_1674636*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
sequential_13/lstm_30/whileá
Fsequential_13/lstm_30/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2H
Fsequential_13/lstm_30/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_13/lstm_30/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_13/lstm_30/while:output:3Osequential_13/lstm_30/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02:
8sequential_13/lstm_30/TensorArrayV2Stack/TensorListStack­
+sequential_13/lstm_30/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_13/lstm_30/strided_slice_3/stack¨
-sequential_13/lstm_30/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_13/lstm_30/strided_slice_3/stack_1¨
-sequential_13/lstm_30/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_30/strided_slice_3/stack_2
%sequential_13/lstm_30/strided_slice_3StridedSliceAsequential_13/lstm_30/TensorArrayV2Stack/TensorListStack:tensor:04sequential_13/lstm_30/strided_slice_3/stack:output:06sequential_13/lstm_30/strided_slice_3/stack_1:output:06sequential_13/lstm_30/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2'
%sequential_13/lstm_30/strided_slice_3¥
&sequential_13/lstm_30/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_13/lstm_30/transpose_1/permþ
!sequential_13/lstm_30/transpose_1	TransposeAsequential_13/lstm_30/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_13/lstm_30/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!sequential_13/lstm_30/transpose_1
sequential_13/lstm_30/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_13/lstm_30/runtime°
!sequential_13/dropout_48/IdentityIdentity%sequential_13/lstm_30/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!sequential_13/dropout_48/Identity
sequential_13/lstm_31/ShapeShape*sequential_13/dropout_48/Identity:output:0*
T0*
_output_shapes
:2
sequential_13/lstm_31/Shape 
)sequential_13/lstm_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_13/lstm_31/strided_slice/stack¤
+sequential_13/lstm_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_31/strided_slice/stack_1¤
+sequential_13/lstm_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_31/strided_slice/stack_2æ
#sequential_13/lstm_31/strided_sliceStridedSlice$sequential_13/lstm_31/Shape:output:02sequential_13/lstm_31/strided_slice/stack:output:04sequential_13/lstm_31/strided_slice/stack_1:output:04sequential_13/lstm_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_13/lstm_31/strided_slice
$sequential_13/lstm_31/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2&
$sequential_13/lstm_31/zeros/packed/1Û
"sequential_13/lstm_31/zeros/packedPack,sequential_13/lstm_31/strided_slice:output:0-sequential_13/lstm_31/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_13/lstm_31/zeros/packed
!sequential_13/lstm_31/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_13/lstm_31/zeros/ConstÎ
sequential_13/lstm_31/zerosFill+sequential_13/lstm_31/zeros/packed:output:0*sequential_13/lstm_31/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
sequential_13/lstm_31/zeros
&sequential_13/lstm_31/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2(
&sequential_13/lstm_31/zeros_1/packed/1á
$sequential_13/lstm_31/zeros_1/packedPack,sequential_13/lstm_31/strided_slice:output:0/sequential_13/lstm_31/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_13/lstm_31/zeros_1/packed
#sequential_13/lstm_31/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_13/lstm_31/zeros_1/ConstÖ
sequential_13/lstm_31/zeros_1Fill-sequential_13/lstm_31/zeros_1/packed:output:0,sequential_13/lstm_31/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
sequential_13/lstm_31/zeros_1¡
$sequential_13/lstm_31/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_13/lstm_31/transpose/permá
sequential_13/lstm_31/transpose	Transpose*sequential_13/dropout_48/Identity:output:0-sequential_13/lstm_31/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
sequential_13/lstm_31/transpose
sequential_13/lstm_31/Shape_1Shape#sequential_13/lstm_31/transpose:y:0*
T0*
_output_shapes
:2
sequential_13/lstm_31/Shape_1¤
+sequential_13/lstm_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_31/strided_slice_1/stack¨
-sequential_13/lstm_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_31/strided_slice_1/stack_1¨
-sequential_13/lstm_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_31/strided_slice_1/stack_2ò
%sequential_13/lstm_31/strided_slice_1StridedSlice&sequential_13/lstm_31/Shape_1:output:04sequential_13/lstm_31/strided_slice_1/stack:output:06sequential_13/lstm_31/strided_slice_1/stack_1:output:06sequential_13/lstm_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_13/lstm_31/strided_slice_1±
1sequential_13/lstm_31/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_13/lstm_31/TensorArrayV2/element_shape
#sequential_13/lstm_31/TensorArrayV2TensorListReserve:sequential_13/lstm_31/TensorArrayV2/element_shape:output:0.sequential_13/lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_13/lstm_31/TensorArrayV2ë
Ksequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2M
Ksequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_13/lstm_31/transpose:y:0Tsequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensor¤
+sequential_13/lstm_31/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_31/strided_slice_2/stack¨
-sequential_13/lstm_31/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_31/strided_slice_2/stack_1¨
-sequential_13/lstm_31/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_31/strided_slice_2/stack_2
%sequential_13/lstm_31/strided_slice_2StridedSlice#sequential_13/lstm_31/transpose:y:04sequential_13/lstm_31/strided_slice_2/stack:output:06sequential_13/lstm_31/strided_slice_2/stack_1:output:06sequential_13/lstm_31/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2'
%sequential_13/lstm_31/strided_slice_2ø
8sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOpReadVariableOpAsequential_13_lstm_31_lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02:
8sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp
)sequential_13/lstm_31/lstm_cell_31/MatMulMatMul.sequential_13/lstm_31/strided_slice_2:output:0@sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2+
)sequential_13/lstm_31/lstm_cell_31/MatMulþ
:sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOpCsequential_13_lstm_31_lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02<
:sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp
+sequential_13/lstm_31/lstm_cell_31/MatMul_1MatMul$sequential_13/lstm_31/zeros:output:0Bsequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2-
+sequential_13/lstm_31/lstm_cell_31/MatMul_1ø
&sequential_13/lstm_31/lstm_cell_31/addAddV23sequential_13/lstm_31/lstm_cell_31/MatMul:product:05sequential_13/lstm_31/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2(
&sequential_13/lstm_31/lstm_cell_31/addö
9sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOpBsequential_13_lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02;
9sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp
*sequential_13/lstm_31/lstm_cell_31/BiasAddBiasAdd*sequential_13/lstm_31/lstm_cell_31/add:z:0Asequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2,
*sequential_13/lstm_31/lstm_cell_31/BiasAddª
2sequential_13/lstm_31/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_13/lstm_31/lstm_cell_31/split/split_dimÏ
(sequential_13/lstm_31/lstm_cell_31/splitSplit;sequential_13/lstm_31/lstm_cell_31/split/split_dim:output:03sequential_13/lstm_31/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2*
(sequential_13/lstm_31/lstm_cell_31/splitÉ
*sequential_13/lstm_31/lstm_cell_31/SigmoidSigmoid1sequential_13/lstm_31/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2,
*sequential_13/lstm_31/lstm_cell_31/SigmoidÍ
,sequential_13/lstm_31/lstm_cell_31/Sigmoid_1Sigmoid1sequential_13/lstm_31/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2.
,sequential_13/lstm_31/lstm_cell_31/Sigmoid_1ä
&sequential_13/lstm_31/lstm_cell_31/mulMul0sequential_13/lstm_31/lstm_cell_31/Sigmoid_1:y:0&sequential_13/lstm_31/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2(
&sequential_13/lstm_31/lstm_cell_31/mulÀ
'sequential_13/lstm_31/lstm_cell_31/ReluRelu1sequential_13/lstm_31/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2)
'sequential_13/lstm_31/lstm_cell_31/Reluõ
(sequential_13/lstm_31/lstm_cell_31/mul_1Mul.sequential_13/lstm_31/lstm_cell_31/Sigmoid:y:05sequential_13/lstm_31/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2*
(sequential_13/lstm_31/lstm_cell_31/mul_1ê
(sequential_13/lstm_31/lstm_cell_31/add_1AddV2*sequential_13/lstm_31/lstm_cell_31/mul:z:0,sequential_13/lstm_31/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2*
(sequential_13/lstm_31/lstm_cell_31/add_1Í
,sequential_13/lstm_31/lstm_cell_31/Sigmoid_2Sigmoid1sequential_13/lstm_31/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2.
,sequential_13/lstm_31/lstm_cell_31/Sigmoid_2¿
)sequential_13/lstm_31/lstm_cell_31/Relu_1Relu,sequential_13/lstm_31/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2+
)sequential_13/lstm_31/lstm_cell_31/Relu_1ù
(sequential_13/lstm_31/lstm_cell_31/mul_2Mul0sequential_13/lstm_31/lstm_cell_31/Sigmoid_2:y:07sequential_13/lstm_31/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2*
(sequential_13/lstm_31/lstm_cell_31/mul_2»
3sequential_13/lstm_31/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   25
3sequential_13/lstm_31/TensorArrayV2_1/element_shape
%sequential_13/lstm_31/TensorArrayV2_1TensorListReserve<sequential_13/lstm_31/TensorArrayV2_1/element_shape:output:0.sequential_13/lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_13/lstm_31/TensorArrayV2_1z
sequential_13/lstm_31/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_13/lstm_31/time«
.sequential_13/lstm_31/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_13/lstm_31/while/maximum_iterations
(sequential_13/lstm_31/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_13/lstm_31/while/loop_counterÛ
sequential_13/lstm_31/whileWhile1sequential_13/lstm_31/while/loop_counter:output:07sequential_13/lstm_31/while/maximum_iterations:output:0#sequential_13/lstm_31/time:output:0.sequential_13/lstm_31/TensorArrayV2_1:handle:0$sequential_13/lstm_31/zeros:output:0&sequential_13/lstm_31/zeros_1:output:0.sequential_13/lstm_31/strided_slice_1:output:0Msequential_13/lstm_31/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_13_lstm_31_lstm_cell_31_matmul_readvariableop_resourceCsequential_13_lstm_31_lstm_cell_31_matmul_1_readvariableop_resourceBsequential_13_lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_13_lstm_31_while_body_1674777*4
cond,R*
(sequential_13_lstm_31_while_cond_1674776*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
sequential_13/lstm_31/whileá
Fsequential_13/lstm_31/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2H
Fsequential_13/lstm_31/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_13/lstm_31/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_13/lstm_31/while:output:3Osequential_13/lstm_31/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02:
8sequential_13/lstm_31/TensorArrayV2Stack/TensorListStack­
+sequential_13/lstm_31/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_13/lstm_31/strided_slice_3/stack¨
-sequential_13/lstm_31/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_13/lstm_31/strided_slice_3/stack_1¨
-sequential_13/lstm_31/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_31/strided_slice_3/stack_2
%sequential_13/lstm_31/strided_slice_3StridedSliceAsequential_13/lstm_31/TensorArrayV2Stack/TensorListStack:tensor:04sequential_13/lstm_31/strided_slice_3/stack:output:06sequential_13/lstm_31/strided_slice_3/stack_1:output:06sequential_13/lstm_31/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2'
%sequential_13/lstm_31/strided_slice_3¥
&sequential_13/lstm_31/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_13/lstm_31/transpose_1/permþ
!sequential_13/lstm_31/transpose_1	TransposeAsequential_13/lstm_31/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_13/lstm_31/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!sequential_13/lstm_31/transpose_1
sequential_13/lstm_31/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_13/lstm_31/runtime°
!sequential_13/dropout_49/IdentityIdentity%sequential_13/lstm_31/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!sequential_13/dropout_49/IdentityÝ
/sequential_13/dense_31/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_31_tensordot_readvariableop_resource* 
_output_shapes
:
´´*
dtype021
/sequential_13/dense_31/Tensordot/ReadVariableOp
%sequential_13/dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_13/dense_31/Tensordot/axes
%sequential_13/dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_13/dense_31/Tensordot/freeª
&sequential_13/dense_31/Tensordot/ShapeShape*sequential_13/dropout_49/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_13/dense_31/Tensordot/Shape¢
.sequential_13/dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_31/Tensordot/GatherV2/axisÄ
)sequential_13/dense_31/Tensordot/GatherV2GatherV2/sequential_13/dense_31/Tensordot/Shape:output:0.sequential_13/dense_31/Tensordot/free:output:07sequential_13/dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_13/dense_31/Tensordot/GatherV2¦
0sequential_13/dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_13/dense_31/Tensordot/GatherV2_1/axisÊ
+sequential_13/dense_31/Tensordot/GatherV2_1GatherV2/sequential_13/dense_31/Tensordot/Shape:output:0.sequential_13/dense_31/Tensordot/axes:output:09sequential_13/dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_13/dense_31/Tensordot/GatherV2_1
&sequential_13/dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_13/dense_31/Tensordot/ConstÜ
%sequential_13/dense_31/Tensordot/ProdProd2sequential_13/dense_31/Tensordot/GatherV2:output:0/sequential_13/dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_13/dense_31/Tensordot/Prod
(sequential_13/dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_13/dense_31/Tensordot/Const_1ä
'sequential_13/dense_31/Tensordot/Prod_1Prod4sequential_13/dense_31/Tensordot/GatherV2_1:output:01sequential_13/dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_13/dense_31/Tensordot/Prod_1
,sequential_13/dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_13/dense_31/Tensordot/concat/axis£
'sequential_13/dense_31/Tensordot/concatConcatV2.sequential_13/dense_31/Tensordot/free:output:0.sequential_13/dense_31/Tensordot/axes:output:05sequential_13/dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_13/dense_31/Tensordot/concatè
&sequential_13/dense_31/Tensordot/stackPack.sequential_13/dense_31/Tensordot/Prod:output:00sequential_13/dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_13/dense_31/Tensordot/stackú
*sequential_13/dense_31/Tensordot/transpose	Transpose*sequential_13/dropout_49/Identity:output:00sequential_13/dense_31/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2,
*sequential_13/dense_31/Tensordot/transposeû
(sequential_13/dense_31/Tensordot/ReshapeReshape.sequential_13/dense_31/Tensordot/transpose:y:0/sequential_13/dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_13/dense_31/Tensordot/Reshapeû
'sequential_13/dense_31/Tensordot/MatMulMatMul1sequential_13/dense_31/Tensordot/Reshape:output:07sequential_13/dense_31/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2)
'sequential_13/dense_31/Tensordot/MatMul
(sequential_13/dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:´2*
(sequential_13/dense_31/Tensordot/Const_2¢
.sequential_13/dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_31/Tensordot/concat_1/axis°
)sequential_13/dense_31/Tensordot/concat_1ConcatV22sequential_13/dense_31/Tensordot/GatherV2:output:01sequential_13/dense_31/Tensordot/Const_2:output:07sequential_13/dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_13/dense_31/Tensordot/concat_1í
 sequential_13/dense_31/TensordotReshape1sequential_13/dense_31/Tensordot/MatMul:product:02sequential_13/dense_31/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 sequential_13/dense_31/TensordotÒ
-sequential_13/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:´*
dtype02/
-sequential_13/dense_31/BiasAdd/ReadVariableOpä
sequential_13/dense_31/BiasAddBiasAdd)sequential_13/dense_31/Tensordot:output:05sequential_13/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
sequential_13/dense_31/BiasAdd¢
sequential_13/dense_31/ReluRelu'sequential_13/dense_31/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
sequential_13/dense_31/Relu´
!sequential_13/dropout_50/IdentityIdentity)sequential_13/dense_31/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!sequential_13/dropout_50/IdentityÜ
/sequential_13/dense_32/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_32_tensordot_readvariableop_resource*
_output_shapes
:	´*
dtype021
/sequential_13/dense_32/Tensordot/ReadVariableOp
%sequential_13/dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_13/dense_32/Tensordot/axes
%sequential_13/dense_32/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_13/dense_32/Tensordot/freeª
&sequential_13/dense_32/Tensordot/ShapeShape*sequential_13/dropout_50/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_13/dense_32/Tensordot/Shape¢
.sequential_13/dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_32/Tensordot/GatherV2/axisÄ
)sequential_13/dense_32/Tensordot/GatherV2GatherV2/sequential_13/dense_32/Tensordot/Shape:output:0.sequential_13/dense_32/Tensordot/free:output:07sequential_13/dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_13/dense_32/Tensordot/GatherV2¦
0sequential_13/dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_13/dense_32/Tensordot/GatherV2_1/axisÊ
+sequential_13/dense_32/Tensordot/GatherV2_1GatherV2/sequential_13/dense_32/Tensordot/Shape:output:0.sequential_13/dense_32/Tensordot/axes:output:09sequential_13/dense_32/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_13/dense_32/Tensordot/GatherV2_1
&sequential_13/dense_32/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_13/dense_32/Tensordot/ConstÜ
%sequential_13/dense_32/Tensordot/ProdProd2sequential_13/dense_32/Tensordot/GatherV2:output:0/sequential_13/dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_13/dense_32/Tensordot/Prod
(sequential_13/dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_13/dense_32/Tensordot/Const_1ä
'sequential_13/dense_32/Tensordot/Prod_1Prod4sequential_13/dense_32/Tensordot/GatherV2_1:output:01sequential_13/dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_13/dense_32/Tensordot/Prod_1
,sequential_13/dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_13/dense_32/Tensordot/concat/axis£
'sequential_13/dense_32/Tensordot/concatConcatV2.sequential_13/dense_32/Tensordot/free:output:0.sequential_13/dense_32/Tensordot/axes:output:05sequential_13/dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_13/dense_32/Tensordot/concatè
&sequential_13/dense_32/Tensordot/stackPack.sequential_13/dense_32/Tensordot/Prod:output:00sequential_13/dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_13/dense_32/Tensordot/stackú
*sequential_13/dense_32/Tensordot/transpose	Transpose*sequential_13/dropout_50/Identity:output:00sequential_13/dense_32/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2,
*sequential_13/dense_32/Tensordot/transposeû
(sequential_13/dense_32/Tensordot/ReshapeReshape.sequential_13/dense_32/Tensordot/transpose:y:0/sequential_13/dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_13/dense_32/Tensordot/Reshapeú
'sequential_13/dense_32/Tensordot/MatMulMatMul1sequential_13/dense_32/Tensordot/Reshape:output:07sequential_13/dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_13/dense_32/Tensordot/MatMul
(sequential_13/dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_13/dense_32/Tensordot/Const_2¢
.sequential_13/dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_32/Tensordot/concat_1/axis°
)sequential_13/dense_32/Tensordot/concat_1ConcatV22sequential_13/dense_32/Tensordot/GatherV2:output:01sequential_13/dense_32/Tensordot/Const_2:output:07sequential_13/dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_13/dense_32/Tensordot/concat_1ì
 sequential_13/dense_32/TensordotReshape1sequential_13/dense_32/Tensordot/MatMul:product:02sequential_13/dense_32/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_13/dense_32/TensordotÑ
-sequential_13/dense_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_13/dense_32/BiasAdd/ReadVariableOpã
sequential_13/dense_32/BiasAddBiasAdd)sequential_13/dense_32/Tensordot:output:05sequential_13/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_13/dense_32/BiasAdd
IdentityIdentity'sequential_13/dense_32/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp.^sequential_13/dense_31/BiasAdd/ReadVariableOp0^sequential_13/dense_31/Tensordot/ReadVariableOp.^sequential_13/dense_32/BiasAdd/ReadVariableOp0^sequential_13/dense_32/Tensordot/ReadVariableOp:^sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp9^sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp;^sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp^sequential_13/lstm_29/while:^sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp9^sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp;^sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp^sequential_13/lstm_30/while:^sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp9^sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp;^sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp^sequential_13/lstm_31/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2^
-sequential_13/dense_31/BiasAdd/ReadVariableOp-sequential_13/dense_31/BiasAdd/ReadVariableOp2b
/sequential_13/dense_31/Tensordot/ReadVariableOp/sequential_13/dense_31/Tensordot/ReadVariableOp2^
-sequential_13/dense_32/BiasAdd/ReadVariableOp-sequential_13/dense_32/BiasAdd/ReadVariableOp2b
/sequential_13/dense_32/Tensordot/ReadVariableOp/sequential_13/dense_32/Tensordot/ReadVariableOp2v
9sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp9sequential_13/lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp2t
8sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp8sequential_13/lstm_29/lstm_cell_29/MatMul/ReadVariableOp2x
:sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp:sequential_13/lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp2:
sequential_13/lstm_29/whilesequential_13/lstm_29/while2v
9sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp9sequential_13/lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp2t
8sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp8sequential_13/lstm_30/lstm_cell_30/MatMul/ReadVariableOp2x
:sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp:sequential_13/lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp2:
sequential_13/lstm_30/whilesequential_13/lstm_30/while2v
9sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp9sequential_13/lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp2t
8sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp8sequential_13/lstm_31/lstm_cell_31/MatMul/ReadVariableOp2x
:sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp:sequential_13/lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp2:
sequential_13/lstm_31/whilesequential_13/lstm_31/while:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_29_input
÷%
î
while_body_1676193
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_31_1676217_0:
´Ð0
while_lstm_cell_31_1676219_0:
´Ð+
while_lstm_cell_31_1676221_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_31_1676217:
´Ð.
while_lstm_cell_31_1676219:
´Ð)
while_lstm_cell_31_1676221:	Ð¢*while/lstm_cell_31/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemè
*while/lstm_cell_31/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_31_1676217_0while_lstm_cell_31_1676219_0while_lstm_cell_31_1676221_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_16761792,
*while/lstm_cell_31/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_31/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_31/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_31/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_31_1676217while_lstm_cell_31_1676217_0":
while_lstm_cell_31_1676219while_lstm_cell_31_1676219_0":
while_lstm_cell_31_1676221while_lstm_cell_31_1676221_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2X
*while/lstm_cell_31/StatefulPartitionedCall*while/lstm_cell_31/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
á
º
)__inference_lstm_31_layer_call_fn_1680497
inputs_0
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_16762622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
inputs/0
+
å
J__inference_sequential_13_layer_call_and_return_conditional_losses_1678076
lstm_29_input"
lstm_29_1678040:	Ð#
lstm_29_1678042:
´Ð
lstm_29_1678044:	Ð#
lstm_30_1678048:
´Ð#
lstm_30_1678050:
´Ð
lstm_30_1678052:	Ð#
lstm_31_1678056:
´Ð#
lstm_31_1678058:
´Ð
lstm_31_1678060:	Ð$
dense_31_1678064:
´´
dense_31_1678066:	´#
dense_32_1678070:	´
dense_32_1678072:
identity¢ dense_31/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢lstm_29/StatefulPartitionedCall¢lstm_30/StatefulPartitionedCall¢lstm_31/StatefulPartitionedCall±
lstm_29/StatefulPartitionedCallStatefulPartitionedCalllstm_29_inputlstm_29_1678040lstm_29_1678042lstm_29_1678044*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_16768592!
lstm_29/StatefulPartitionedCall
dropout_47/PartitionedCallPartitionedCall(lstm_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_16768722
dropout_47/PartitionedCallÇ
lstm_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_47/PartitionedCall:output:0lstm_30_1678048lstm_30_1678050lstm_30_1678052*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_16770162!
lstm_30/StatefulPartitionedCall
dropout_48/PartitionedCallPartitionedCall(lstm_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_16770292
dropout_48/PartitionedCallÇ
lstm_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_48/PartitionedCall:output:0lstm_31_1678056lstm_31_1678058lstm_31_1678060*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_16771732!
lstm_31/StatefulPartitionedCall
dropout_49/PartitionedCallPartitionedCall(lstm_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_16771862
dropout_49/PartitionedCall¹
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_49/PartitionedCall:output:0dense_31_1678064dense_31_1678066*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_16772192"
 dense_31/StatefulPartitionedCall
dropout_50/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_16772302
dropout_50/PartitionedCall¸
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0dense_32_1678070dense_32_1678072*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_16772622"
 dense_32/StatefulPartitionedCall
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityú
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall ^lstm_30/StatefulPartitionedCall ^lstm_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall2B
lstm_30/StatefulPartitionedCalllstm_30/StatefulPartitionedCall2B
lstm_31/StatefulPartitionedCalllstm_31/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_29_input
Þ
È
while_cond_1679302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1679302___redundant_placeholder05
1while_while_cond_1679302___redundant_placeholder15
1while_while_cond_1679302___redundant_placeholder25
1while_while_cond_1679302___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:

e
G__inference_dropout_48_layer_call_and_return_conditional_losses_1680474

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Þ
È
while_cond_1680874
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1680874___redundant_placeholder05
1while_while_cond_1680874___redundant_placeholder15
1while_while_cond_1680874___redundant_placeholder25
1while_while_cond_1680874___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:

e
G__inference_dropout_48_layer_call_and_return_conditional_losses_1677029

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Æ1
ù
J__inference_sequential_13_layer_call_and_return_conditional_losses_1678115
lstm_29_input"
lstm_29_1678079:	Ð#
lstm_29_1678081:
´Ð
lstm_29_1678083:	Ð#
lstm_30_1678087:
´Ð#
lstm_30_1678089:
´Ð
lstm_30_1678091:	Ð#
lstm_31_1678095:
´Ð#
lstm_31_1678097:
´Ð
lstm_31_1678099:	Ð$
dense_31_1678103:
´´
dense_31_1678105:	´#
dense_32_1678109:	´
dense_32_1678111:
identity¢ dense_31/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢"dropout_47/StatefulPartitionedCall¢"dropout_48/StatefulPartitionedCall¢"dropout_49/StatefulPartitionedCall¢"dropout_50/StatefulPartitionedCall¢lstm_29/StatefulPartitionedCall¢lstm_30/StatefulPartitionedCall¢lstm_31/StatefulPartitionedCall±
lstm_29/StatefulPartitionedCallStatefulPartitionedCalllstm_29_inputlstm_29_1678079lstm_29_1678081lstm_29_1678083*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_16778962!
lstm_29/StatefulPartitionedCall
"dropout_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_16777372$
"dropout_47/StatefulPartitionedCallÏ
lstm_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_47/StatefulPartitionedCall:output:0lstm_30_1678087lstm_30_1678089lstm_30_1678091*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_16777082!
lstm_30/StatefulPartitionedCall¿
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall(lstm_30/StatefulPartitionedCall:output:0#^dropout_47/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_16775492$
"dropout_48/StatefulPartitionedCallÏ
lstm_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_48/StatefulPartitionedCall:output:0lstm_31_1678095lstm_31_1678097lstm_31_1678099*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_16775202!
lstm_31/StatefulPartitionedCall¿
"dropout_49/StatefulPartitionedCallStatefulPartitionedCall(lstm_31/StatefulPartitionedCall:output:0#^dropout_48/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_16773612$
"dropout_49/StatefulPartitionedCallÁ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_49/StatefulPartitionedCall:output:0dense_31_1678103dense_31_1678105*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_16772192"
 dense_31/StatefulPartitionedCallÀ
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0#^dropout_49/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_16773282$
"dropout_50/StatefulPartitionedCallÀ
 dense_32/StatefulPartitionedCallStatefulPartitionedCall+dropout_50/StatefulPartitionedCall:output:0dense_32_1678109dense_32_1678111*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_16772622"
 dense_32/StatefulPartitionedCall
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall#^dropout_47/StatefulPartitionedCall#^dropout_48/StatefulPartitionedCall#^dropout_49/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall ^lstm_30/StatefulPartitionedCall ^lstm_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2H
"dropout_47/StatefulPartitionedCall"dropout_47/StatefulPartitionedCall2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall2H
"dropout_49/StatefulPartitionedCall"dropout_49/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall2B
lstm_30/StatefulPartitionedCalllstm_30/StatefulPartitionedCall2B
lstm_31/StatefulPartitionedCalllstm_31/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_29_input


I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1681333

inputs
states_0
states_11
matmul_readvariableop_resource:	Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
Ö
f
G__inference_dropout_50_layer_call_and_return_conditional_losses_1677328

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Ö
f
G__inference_dropout_48_layer_call_and_return_conditional_losses_1677549

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs


I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1676179

inputs

states
states_12
matmul_readvariableop_resource:
´Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates

½
J__inference_sequential_13_layer_call_and_return_conditional_losses_1679200

inputsF
3lstm_29_lstm_cell_29_matmul_readvariableop_resource:	ÐI
5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐC
4lstm_29_lstm_cell_29_biasadd_readvariableop_resource:	ÐG
3lstm_30_lstm_cell_30_matmul_readvariableop_resource:
´ÐI
5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐC
4lstm_30_lstm_cell_30_biasadd_readvariableop_resource:	ÐG
3lstm_31_lstm_cell_31_matmul_readvariableop_resource:
´ÐI
5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐC
4lstm_31_lstm_cell_31_biasadd_readvariableop_resource:	Ð>
*dense_31_tensordot_readvariableop_resource:
´´7
(dense_31_biasadd_readvariableop_resource:	´=
*dense_32_tensordot_readvariableop_resource:	´6
(dense_32_biasadd_readvariableop_resource:
identity¢dense_31/BiasAdd/ReadVariableOp¢!dense_31/Tensordot/ReadVariableOp¢dense_32/BiasAdd/ReadVariableOp¢!dense_32/Tensordot/ReadVariableOp¢+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp¢*lstm_29/lstm_cell_29/MatMul/ReadVariableOp¢,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp¢lstm_29/while¢+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp¢*lstm_30/lstm_cell_30/MatMul/ReadVariableOp¢,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp¢lstm_30/while¢+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp¢*lstm_31/lstm_cell_31/MatMul/ReadVariableOp¢,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp¢lstm_31/whileT
lstm_29/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_29/Shape
lstm_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice/stack
lstm_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_29/strided_slice/stack_1
lstm_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_29/strided_slice/stack_2
lstm_29/strided_sliceStridedSlicelstm_29/Shape:output:0$lstm_29/strided_slice/stack:output:0&lstm_29/strided_slice/stack_1:output:0&lstm_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_29/strided_slices
lstm_29/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_29/zeros/packed/1£
lstm_29/zeros/packedPacklstm_29/strided_slice:output:0lstm_29/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_29/zeros/packedo
lstm_29/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/zeros/Const
lstm_29/zerosFilllstm_29/zeros/packed:output:0lstm_29/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/zerosw
lstm_29/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_29/zeros_1/packed/1©
lstm_29/zeros_1/packedPacklstm_29/strided_slice:output:0!lstm_29/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_29/zeros_1/packeds
lstm_29/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/zeros_1/Const
lstm_29/zeros_1Filllstm_29/zeros_1/packed:output:0lstm_29/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/zeros_1
lstm_29/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_29/transpose/perm
lstm_29/transpose	Transposeinputslstm_29/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_29/transposeg
lstm_29/Shape_1Shapelstm_29/transpose:y:0*
T0*
_output_shapes
:2
lstm_29/Shape_1
lstm_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice_1/stack
lstm_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_1/stack_1
lstm_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_1/stack_2
lstm_29/strided_slice_1StridedSlicelstm_29/Shape_1:output:0&lstm_29/strided_slice_1/stack:output:0(lstm_29/strided_slice_1/stack_1:output:0(lstm_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_29/strided_slice_1
#lstm_29/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_29/TensorArrayV2/element_shapeÒ
lstm_29/TensorArrayV2TensorListReserve,lstm_29/TensorArrayV2/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_29/TensorArrayV2Ï
=lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_29/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_29/transpose:y:0Flstm_29/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_29/TensorArrayUnstack/TensorListFromTensor
lstm_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_29/strided_slice_2/stack
lstm_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_2/stack_1
lstm_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_2/stack_2¬
lstm_29/strided_slice_2StridedSlicelstm_29/transpose:y:0&lstm_29/strided_slice_2/stack:output:0(lstm_29/strided_slice_2/stack_1:output:0(lstm_29/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_29/strided_slice_2Í
*lstm_29/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3lstm_29_lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02,
*lstm_29/lstm_cell_29/MatMul/ReadVariableOpÍ
lstm_29/lstm_cell_29/MatMulMatMul lstm_29/strided_slice_2:output:02lstm_29/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_29/lstm_cell_29/MatMulÔ
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02.
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOpÉ
lstm_29/lstm_cell_29/MatMul_1MatMullstm_29/zeros:output:04lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_29/lstm_cell_29/MatMul_1À
lstm_29/lstm_cell_29/addAddV2%lstm_29/lstm_cell_29/MatMul:product:0'lstm_29/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_29/lstm_cell_29/addÌ
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02-
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOpÍ
lstm_29/lstm_cell_29/BiasAddBiasAddlstm_29/lstm_cell_29/add:z:03lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_29/lstm_cell_29/BiasAdd
$lstm_29/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_29/lstm_cell_29/split/split_dim
lstm_29/lstm_cell_29/splitSplit-lstm_29/lstm_cell_29/split/split_dim:output:0%lstm_29/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_29/lstm_cell_29/split
lstm_29/lstm_cell_29/SigmoidSigmoid#lstm_29/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/Sigmoid£
lstm_29/lstm_cell_29/Sigmoid_1Sigmoid#lstm_29/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_29/lstm_cell_29/Sigmoid_1¬
lstm_29/lstm_cell_29/mulMul"lstm_29/lstm_cell_29/Sigmoid_1:y:0lstm_29/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/mul
lstm_29/lstm_cell_29/ReluRelu#lstm_29/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/Relu½
lstm_29/lstm_cell_29/mul_1Mul lstm_29/lstm_cell_29/Sigmoid:y:0'lstm_29/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/mul_1²
lstm_29/lstm_cell_29/add_1AddV2lstm_29/lstm_cell_29/mul:z:0lstm_29/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/add_1£
lstm_29/lstm_cell_29/Sigmoid_2Sigmoid#lstm_29/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_29/lstm_cell_29/Sigmoid_2
lstm_29/lstm_cell_29/Relu_1Relulstm_29/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/Relu_1Á
lstm_29/lstm_cell_29/mul_2Mul"lstm_29/lstm_cell_29/Sigmoid_2:y:0)lstm_29/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/lstm_cell_29/mul_2
%lstm_29/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2'
%lstm_29/TensorArrayV2_1/element_shapeØ
lstm_29/TensorArrayV2_1TensorListReserve.lstm_29/TensorArrayV2_1/element_shape:output:0 lstm_29/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_29/TensorArrayV2_1^
lstm_29/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_29/time
 lstm_29/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_29/while/maximum_iterationsz
lstm_29/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_29/while/loop_counter
lstm_29/whileWhile#lstm_29/while/loop_counter:output:0)lstm_29/while/maximum_iterations:output:0lstm_29/time:output:0 lstm_29/TensorArrayV2_1:handle:0lstm_29/zeros:output:0lstm_29/zeros_1:output:0 lstm_29/strided_slice_1:output:0?lstm_29/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_29_lstm_cell_29_matmul_readvariableop_resource5lstm_29_lstm_cell_29_matmul_1_readvariableop_resource4lstm_29_lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_29_while_body_1678753*&
condR
lstm_29_while_cond_1678752*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
lstm_29/whileÅ
8lstm_29/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2:
8lstm_29/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_29/TensorArrayV2Stack/TensorListStackTensorListStacklstm_29/while:output:3Alstm_29/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02,
*lstm_29/TensorArrayV2Stack/TensorListStack
lstm_29/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_29/strided_slice_3/stack
lstm_29/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_29/strided_slice_3/stack_1
lstm_29/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_29/strided_slice_3/stack_2Ë
lstm_29/strided_slice_3StridedSlice3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_29/strided_slice_3/stack:output:0(lstm_29/strided_slice_3/stack_1:output:0(lstm_29/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
lstm_29/strided_slice_3
lstm_29/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_29/transpose_1/permÆ
lstm_29/transpose_1	Transpose3lstm_29/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_29/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_29/transpose_1v
lstm_29/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_29/runtimey
dropout_47/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_47/dropout/Constª
dropout_47/dropout/MulMullstm_29/transpose_1:y:0!dropout_47/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_47/dropout/Mul{
dropout_47/dropout/ShapeShapelstm_29/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_47/dropout/ShapeÚ
/dropout_47/dropout/random_uniform/RandomUniformRandomUniform!dropout_47/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype021
/dropout_47/dropout/random_uniform/RandomUniform
!dropout_47/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_47/dropout/GreaterEqual/yï
dropout_47/dropout/GreaterEqualGreaterEqual8dropout_47/dropout/random_uniform/RandomUniform:output:0*dropout_47/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
dropout_47/dropout/GreaterEqual¥
dropout_47/dropout/CastCast#dropout_47/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_47/dropout/Cast«
dropout_47/dropout/Mul_1Muldropout_47/dropout/Mul:z:0dropout_47/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_47/dropout/Mul_1j
lstm_30/ShapeShapedropout_47/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_30/Shape
lstm_30/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice/stack
lstm_30/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_30/strided_slice/stack_1
lstm_30/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_30/strided_slice/stack_2
lstm_30/strided_sliceStridedSlicelstm_30/Shape:output:0$lstm_30/strided_slice/stack:output:0&lstm_30/strided_slice/stack_1:output:0&lstm_30/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_30/strided_slices
lstm_30/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_30/zeros/packed/1£
lstm_30/zeros/packedPacklstm_30/strided_slice:output:0lstm_30/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_30/zeros/packedo
lstm_30/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/zeros/Const
lstm_30/zerosFilllstm_30/zeros/packed:output:0lstm_30/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/zerosw
lstm_30/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_30/zeros_1/packed/1©
lstm_30/zeros_1/packedPacklstm_30/strided_slice:output:0!lstm_30/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_30/zeros_1/packeds
lstm_30/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/zeros_1/Const
lstm_30/zeros_1Filllstm_30/zeros_1/packed:output:0lstm_30/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/zeros_1
lstm_30/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_30/transpose/perm©
lstm_30/transpose	Transposedropout_47/dropout/Mul_1:z:0lstm_30/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/transposeg
lstm_30/Shape_1Shapelstm_30/transpose:y:0*
T0*
_output_shapes
:2
lstm_30/Shape_1
lstm_30/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice_1/stack
lstm_30/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_1/stack_1
lstm_30/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_1/stack_2
lstm_30/strided_slice_1StridedSlicelstm_30/Shape_1:output:0&lstm_30/strided_slice_1/stack:output:0(lstm_30/strided_slice_1/stack_1:output:0(lstm_30/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_30/strided_slice_1
#lstm_30/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_30/TensorArrayV2/element_shapeÒ
lstm_30/TensorArrayV2TensorListReserve,lstm_30/TensorArrayV2/element_shape:output:0 lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_30/TensorArrayV2Ï
=lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2?
=lstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_30/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_30/transpose:y:0Flstm_30/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_30/TensorArrayUnstack/TensorListFromTensor
lstm_30/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_30/strided_slice_2/stack
lstm_30/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_2/stack_1
lstm_30/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_2/stack_2­
lstm_30/strided_slice_2StridedSlicelstm_30/transpose:y:0&lstm_30/strided_slice_2/stack:output:0(lstm_30/strided_slice_2/stack_1:output:0(lstm_30/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
lstm_30/strided_slice_2Î
*lstm_30/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3lstm_30_lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02,
*lstm_30/lstm_cell_30/MatMul/ReadVariableOpÍ
lstm_30/lstm_cell_30/MatMulMatMul lstm_30/strided_slice_2:output:02lstm_30/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_30/lstm_cell_30/MatMulÔ
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02.
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOpÉ
lstm_30/lstm_cell_30/MatMul_1MatMullstm_30/zeros:output:04lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_30/lstm_cell_30/MatMul_1À
lstm_30/lstm_cell_30/addAddV2%lstm_30/lstm_cell_30/MatMul:product:0'lstm_30/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_30/lstm_cell_30/addÌ
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02-
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOpÍ
lstm_30/lstm_cell_30/BiasAddBiasAddlstm_30/lstm_cell_30/add:z:03lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_30/lstm_cell_30/BiasAdd
$lstm_30/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_30/lstm_cell_30/split/split_dim
lstm_30/lstm_cell_30/splitSplit-lstm_30/lstm_cell_30/split/split_dim:output:0%lstm_30/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_30/lstm_cell_30/split
lstm_30/lstm_cell_30/SigmoidSigmoid#lstm_30/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/Sigmoid£
lstm_30/lstm_cell_30/Sigmoid_1Sigmoid#lstm_30/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_30/lstm_cell_30/Sigmoid_1¬
lstm_30/lstm_cell_30/mulMul"lstm_30/lstm_cell_30/Sigmoid_1:y:0lstm_30/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/mul
lstm_30/lstm_cell_30/ReluRelu#lstm_30/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/Relu½
lstm_30/lstm_cell_30/mul_1Mul lstm_30/lstm_cell_30/Sigmoid:y:0'lstm_30/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/mul_1²
lstm_30/lstm_cell_30/add_1AddV2lstm_30/lstm_cell_30/mul:z:0lstm_30/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/add_1£
lstm_30/lstm_cell_30/Sigmoid_2Sigmoid#lstm_30/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_30/lstm_cell_30/Sigmoid_2
lstm_30/lstm_cell_30/Relu_1Relulstm_30/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/Relu_1Á
lstm_30/lstm_cell_30/mul_2Mul"lstm_30/lstm_cell_30/Sigmoid_2:y:0)lstm_30/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/lstm_cell_30/mul_2
%lstm_30/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2'
%lstm_30/TensorArrayV2_1/element_shapeØ
lstm_30/TensorArrayV2_1TensorListReserve.lstm_30/TensorArrayV2_1/element_shape:output:0 lstm_30/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_30/TensorArrayV2_1^
lstm_30/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_30/time
 lstm_30/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_30/while/maximum_iterationsz
lstm_30/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_30/while/loop_counter
lstm_30/whileWhile#lstm_30/while/loop_counter:output:0)lstm_30/while/maximum_iterations:output:0lstm_30/time:output:0 lstm_30/TensorArrayV2_1:handle:0lstm_30/zeros:output:0lstm_30/zeros_1:output:0 lstm_30/strided_slice_1:output:0?lstm_30/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_30_lstm_cell_30_matmul_readvariableop_resource5lstm_30_lstm_cell_30_matmul_1_readvariableop_resource4lstm_30_lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_30_while_body_1678900*&
condR
lstm_30_while_cond_1678899*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
lstm_30/whileÅ
8lstm_30/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2:
8lstm_30/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_30/TensorArrayV2Stack/TensorListStackTensorListStacklstm_30/while:output:3Alstm_30/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02,
*lstm_30/TensorArrayV2Stack/TensorListStack
lstm_30/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_30/strided_slice_3/stack
lstm_30/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_30/strided_slice_3/stack_1
lstm_30/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_30/strided_slice_3/stack_2Ë
lstm_30/strided_slice_3StridedSlice3lstm_30/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_30/strided_slice_3/stack:output:0(lstm_30/strided_slice_3/stack_1:output:0(lstm_30/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
lstm_30/strided_slice_3
lstm_30/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_30/transpose_1/permÆ
lstm_30/transpose_1	Transpose3lstm_30/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_30/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/transpose_1v
lstm_30/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_30/runtimey
dropout_48/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_48/dropout/Constª
dropout_48/dropout/MulMullstm_30/transpose_1:y:0!dropout_48/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_48/dropout/Mul{
dropout_48/dropout/ShapeShapelstm_30/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_48/dropout/ShapeÚ
/dropout_48/dropout/random_uniform/RandomUniformRandomUniform!dropout_48/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype021
/dropout_48/dropout/random_uniform/RandomUniform
!dropout_48/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_48/dropout/GreaterEqual/yï
dropout_48/dropout/GreaterEqualGreaterEqual8dropout_48/dropout/random_uniform/RandomUniform:output:0*dropout_48/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
dropout_48/dropout/GreaterEqual¥
dropout_48/dropout/CastCast#dropout_48/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_48/dropout/Cast«
dropout_48/dropout/Mul_1Muldropout_48/dropout/Mul:z:0dropout_48/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_48/dropout/Mul_1j
lstm_31/ShapeShapedropout_48/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_31/Shape
lstm_31/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice/stack
lstm_31/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_31/strided_slice/stack_1
lstm_31/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_31/strided_slice/stack_2
lstm_31/strided_sliceStridedSlicelstm_31/Shape:output:0$lstm_31/strided_slice/stack:output:0&lstm_31/strided_slice/stack_1:output:0&lstm_31/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_31/strided_slices
lstm_31/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_31/zeros/packed/1£
lstm_31/zeros/packedPacklstm_31/strided_slice:output:0lstm_31/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_31/zeros/packedo
lstm_31/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/zeros/Const
lstm_31/zerosFilllstm_31/zeros/packed:output:0lstm_31/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/zerosw
lstm_31/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
lstm_31/zeros_1/packed/1©
lstm_31/zeros_1/packedPacklstm_31/strided_slice:output:0!lstm_31/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_31/zeros_1/packeds
lstm_31/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/zeros_1/Const
lstm_31/zeros_1Filllstm_31/zeros_1/packed:output:0lstm_31/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/zeros_1
lstm_31/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_31/transpose/perm©
lstm_31/transpose	Transposedropout_48/dropout/Mul_1:z:0lstm_31/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/transposeg
lstm_31/Shape_1Shapelstm_31/transpose:y:0*
T0*
_output_shapes
:2
lstm_31/Shape_1
lstm_31/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice_1/stack
lstm_31/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_1/stack_1
lstm_31/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_1/stack_2
lstm_31/strided_slice_1StridedSlicelstm_31/Shape_1:output:0&lstm_31/strided_slice_1/stack:output:0(lstm_31/strided_slice_1/stack_1:output:0(lstm_31/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_31/strided_slice_1
#lstm_31/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_31/TensorArrayV2/element_shapeÒ
lstm_31/TensorArrayV2TensorListReserve,lstm_31/TensorArrayV2/element_shape:output:0 lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_31/TensorArrayV2Ï
=lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2?
=lstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_31/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_31/transpose:y:0Flstm_31/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_31/TensorArrayUnstack/TensorListFromTensor
lstm_31/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_31/strided_slice_2/stack
lstm_31/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_2/stack_1
lstm_31/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_2/stack_2­
lstm_31/strided_slice_2StridedSlicelstm_31/transpose:y:0&lstm_31/strided_slice_2/stack:output:0(lstm_31/strided_slice_2/stack_1:output:0(lstm_31/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
lstm_31/strided_slice_2Î
*lstm_31/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3lstm_31_lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02,
*lstm_31/lstm_cell_31/MatMul/ReadVariableOpÍ
lstm_31/lstm_cell_31/MatMulMatMul lstm_31/strided_slice_2:output:02lstm_31/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_31/lstm_cell_31/MatMulÔ
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02.
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOpÉ
lstm_31/lstm_cell_31/MatMul_1MatMullstm_31/zeros:output:04lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_31/lstm_cell_31/MatMul_1À
lstm_31/lstm_cell_31/addAddV2%lstm_31/lstm_cell_31/MatMul:product:0'lstm_31/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_31/lstm_cell_31/addÌ
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02-
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOpÍ
lstm_31/lstm_cell_31/BiasAddBiasAddlstm_31/lstm_cell_31/add:z:03lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_31/lstm_cell_31/BiasAdd
$lstm_31/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_31/lstm_cell_31/split/split_dim
lstm_31/lstm_cell_31/splitSplit-lstm_31/lstm_cell_31/split/split_dim:output:0%lstm_31/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_31/lstm_cell_31/split
lstm_31/lstm_cell_31/SigmoidSigmoid#lstm_31/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/Sigmoid£
lstm_31/lstm_cell_31/Sigmoid_1Sigmoid#lstm_31/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_31/lstm_cell_31/Sigmoid_1¬
lstm_31/lstm_cell_31/mulMul"lstm_31/lstm_cell_31/Sigmoid_1:y:0lstm_31/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/mul
lstm_31/lstm_cell_31/ReluRelu#lstm_31/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/Relu½
lstm_31/lstm_cell_31/mul_1Mul lstm_31/lstm_cell_31/Sigmoid:y:0'lstm_31/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/mul_1²
lstm_31/lstm_cell_31/add_1AddV2lstm_31/lstm_cell_31/mul:z:0lstm_31/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/add_1£
lstm_31/lstm_cell_31/Sigmoid_2Sigmoid#lstm_31/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_31/lstm_cell_31/Sigmoid_2
lstm_31/lstm_cell_31/Relu_1Relulstm_31/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/Relu_1Á
lstm_31/lstm_cell_31/mul_2Mul"lstm_31/lstm_cell_31/Sigmoid_2:y:0)lstm_31/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/lstm_cell_31/mul_2
%lstm_31/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2'
%lstm_31/TensorArrayV2_1/element_shapeØ
lstm_31/TensorArrayV2_1TensorListReserve.lstm_31/TensorArrayV2_1/element_shape:output:0 lstm_31/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_31/TensorArrayV2_1^
lstm_31/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_31/time
 lstm_31/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_31/while/maximum_iterationsz
lstm_31/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_31/while/loop_counter
lstm_31/whileWhile#lstm_31/while/loop_counter:output:0)lstm_31/while/maximum_iterations:output:0lstm_31/time:output:0 lstm_31/TensorArrayV2_1:handle:0lstm_31/zeros:output:0lstm_31/zeros_1:output:0 lstm_31/strided_slice_1:output:0?lstm_31/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_31_lstm_cell_31_matmul_readvariableop_resource5lstm_31_lstm_cell_31_matmul_1_readvariableop_resource4lstm_31_lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_31_while_body_1679047*&
condR
lstm_31_while_cond_1679046*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
lstm_31/whileÅ
8lstm_31/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2:
8lstm_31/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_31/TensorArrayV2Stack/TensorListStackTensorListStacklstm_31/while:output:3Alstm_31/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02,
*lstm_31/TensorArrayV2Stack/TensorListStack
lstm_31/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_31/strided_slice_3/stack
lstm_31/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_31/strided_slice_3/stack_1
lstm_31/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_31/strided_slice_3/stack_2Ë
lstm_31/strided_slice_3StridedSlice3lstm_31/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_31/strided_slice_3/stack:output:0(lstm_31/strided_slice_3/stack_1:output:0(lstm_31/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
lstm_31/strided_slice_3
lstm_31/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_31/transpose_1/permÆ
lstm_31/transpose_1	Transpose3lstm_31/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_31/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/transpose_1v
lstm_31/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_31/runtimey
dropout_49/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_49/dropout/Constª
dropout_49/dropout/MulMullstm_31/transpose_1:y:0!dropout_49/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_49/dropout/Mul{
dropout_49/dropout/ShapeShapelstm_31/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_49/dropout/ShapeÚ
/dropout_49/dropout/random_uniform/RandomUniformRandomUniform!dropout_49/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype021
/dropout_49/dropout/random_uniform/RandomUniform
!dropout_49/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_49/dropout/GreaterEqual/yï
dropout_49/dropout/GreaterEqualGreaterEqual8dropout_49/dropout/random_uniform/RandomUniform:output:0*dropout_49/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
dropout_49/dropout/GreaterEqual¥
dropout_49/dropout/CastCast#dropout_49/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_49/dropout/Cast«
dropout_49/dropout/Mul_1Muldropout_49/dropout/Mul:z:0dropout_49/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_49/dropout/Mul_1³
!dense_31/Tensordot/ReadVariableOpReadVariableOp*dense_31_tensordot_readvariableop_resource* 
_output_shapes
:
´´*
dtype02#
!dense_31/Tensordot/ReadVariableOp|
dense_31/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_31/Tensordot/axes
dense_31/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_31/Tensordot/free
dense_31/Tensordot/ShapeShapedropout_49/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_31/Tensordot/Shape
 dense_31/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/GatherV2/axisþ
dense_31/Tensordot/GatherV2GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/free:output:0)dense_31/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2
"dense_31/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_31/Tensordot/GatherV2_1/axis
dense_31/Tensordot/GatherV2_1GatherV2!dense_31/Tensordot/Shape:output:0 dense_31/Tensordot/axes:output:0+dense_31/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_31/Tensordot/GatherV2_1~
dense_31/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const¤
dense_31/Tensordot/ProdProd$dense_31/Tensordot/GatherV2:output:0!dense_31/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod
dense_31/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_31/Tensordot/Const_1¬
dense_31/Tensordot/Prod_1Prod&dense_31/Tensordot/GatherV2_1:output:0#dense_31/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_31/Tensordot/Prod_1
dense_31/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_31/Tensordot/concat/axisÝ
dense_31/Tensordot/concatConcatV2 dense_31/Tensordot/free:output:0 dense_31/Tensordot/axes:output:0'dense_31/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat°
dense_31/Tensordot/stackPack dense_31/Tensordot/Prod:output:0"dense_31/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/stackÂ
dense_31/Tensordot/transpose	Transposedropout_49/dropout/Mul_1:z:0"dense_31/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_31/Tensordot/transposeÃ
dense_31/Tensordot/ReshapeReshape dense_31/Tensordot/transpose:y:0!dense_31/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_31/Tensordot/ReshapeÃ
dense_31/Tensordot/MatMulMatMul#dense_31/Tensordot/Reshape:output:0)dense_31/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_31/Tensordot/MatMul
dense_31/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:´2
dense_31/Tensordot/Const_2
 dense_31/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_31/Tensordot/concat_1/axisê
dense_31/Tensordot/concat_1ConcatV2$dense_31/Tensordot/GatherV2:output:0#dense_31/Tensordot/Const_2:output:0)dense_31/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_31/Tensordot/concat_1µ
dense_31/TensordotReshape#dense_31/Tensordot/MatMul:product:0$dense_31/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_31/Tensordot¨
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:´*
dtype02!
dense_31/BiasAdd/ReadVariableOp¬
dense_31/BiasAddBiasAdddense_31/Tensordot:output:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_31/BiasAddx
dense_31/ReluReludense_31/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_31/Reluy
dropout_50/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_50/dropout/Const®
dropout_50/dropout/MulMuldense_31/Relu:activations:0!dropout_50/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_50/dropout/Mul
dropout_50/dropout/ShapeShapedense_31/Relu:activations:0*
T0*
_output_shapes
:2
dropout_50/dropout/ShapeÚ
/dropout_50/dropout/random_uniform/RandomUniformRandomUniform!dropout_50/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype021
/dropout_50/dropout/random_uniform/RandomUniform
!dropout_50/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_50/dropout/GreaterEqual/yï
dropout_50/dropout/GreaterEqualGreaterEqual8dropout_50/dropout/random_uniform/RandomUniform:output:0*dropout_50/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
dropout_50/dropout/GreaterEqual¥
dropout_50/dropout/CastCast#dropout_50/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_50/dropout/Cast«
dropout_50/dropout/Mul_1Muldropout_50/dropout/Mul:z:0dropout_50/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout_50/dropout/Mul_1²
!dense_32/Tensordot/ReadVariableOpReadVariableOp*dense_32_tensordot_readvariableop_resource*
_output_shapes
:	´*
dtype02#
!dense_32/Tensordot/ReadVariableOp|
dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_32/Tensordot/axes
dense_32/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_32/Tensordot/free
dense_32/Tensordot/ShapeShapedropout_50/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_32/Tensordot/Shape
 dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/GatherV2/axisþ
dense_32/Tensordot/GatherV2GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/free:output:0)dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2
"dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_32/Tensordot/GatherV2_1/axis
dense_32/Tensordot/GatherV2_1GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/axes:output:0+dense_32/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2_1~
dense_32/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const¤
dense_32/Tensordot/ProdProd$dense_32/Tensordot/GatherV2:output:0!dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod
dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_1¬
dense_32/Tensordot/Prod_1Prod&dense_32/Tensordot/GatherV2_1:output:0#dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod_1
dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_32/Tensordot/concat/axisÝ
dense_32/Tensordot/concatConcatV2 dense_32/Tensordot/free:output:0 dense_32/Tensordot/axes:output:0'dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat°
dense_32/Tensordot/stackPack dense_32/Tensordot/Prod:output:0"dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/stackÂ
dense_32/Tensordot/transpose	Transposedropout_50/dropout/Mul_1:z:0"dense_32/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dense_32/Tensordot/transposeÃ
dense_32/Tensordot/ReshapeReshape dense_32/Tensordot/transpose:y:0!dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_32/Tensordot/ReshapeÂ
dense_32/Tensordot/MatMulMatMul#dense_32/Tensordot/Reshape:output:0)dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_32/Tensordot/MatMul
dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_32/Tensordot/Const_2
 dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/concat_1/axisê
dense_32/Tensordot/concat_1ConcatV2$dense_32/Tensordot/GatherV2:output:0#dense_32/Tensordot/Const_2:output:0)dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat_1´
dense_32/TensordotReshape#dense_32/Tensordot/MatMul:product:0$dense_32/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_32/Tensordot§
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_32/BiasAdd/ReadVariableOp«
dense_32/BiasAddBiasAdddense_32/Tensordot:output:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_32/BiasAddx
IdentityIdentitydense_32/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¨
NoOpNoOp ^dense_31/BiasAdd/ReadVariableOp"^dense_31/Tensordot/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp"^dense_32/Tensordot/ReadVariableOp,^lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp+^lstm_29/lstm_cell_29/MatMul/ReadVariableOp-^lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp^lstm_29/while,^lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp+^lstm_30/lstm_cell_30/MatMul/ReadVariableOp-^lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp^lstm_30/while,^lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp+^lstm_31/lstm_cell_31/MatMul/ReadVariableOp-^lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp^lstm_31/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2F
!dense_31/Tensordot/ReadVariableOp!dense_31/Tensordot/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2F
!dense_32/Tensordot/ReadVariableOp!dense_32/Tensordot/ReadVariableOp2Z
+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp+lstm_29/lstm_cell_29/BiasAdd/ReadVariableOp2X
*lstm_29/lstm_cell_29/MatMul/ReadVariableOp*lstm_29/lstm_cell_29/MatMul/ReadVariableOp2\
,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp,lstm_29/lstm_cell_29/MatMul_1/ReadVariableOp2
lstm_29/whilelstm_29/while2Z
+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp+lstm_30/lstm_cell_30/BiasAdd/ReadVariableOp2X
*lstm_30/lstm_cell_30/MatMul/ReadVariableOp*lstm_30/lstm_cell_30/MatMul/ReadVariableOp2\
,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp,lstm_30/lstm_cell_30/MatMul_1/ReadVariableOp2
lstm_30/whilelstm_30/while2Z
+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp+lstm_31/lstm_cell_31/BiasAdd/ReadVariableOp2X
*lstm_31/lstm_cell_31/MatMul/ReadVariableOp*lstm_31/lstm_cell_31/MatMul/ReadVariableOp2\
,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp,lstm_31/lstm_cell_31/MatMul_1/ReadVariableOp2
lstm_31/whilelstm_31/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³?
Õ
while_body_1680232
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_30_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_30/BiasAdd/ReadVariableOp¢(while/lstm_cell_30/MatMul/ReadVariableOp¢*while/lstm_cell_30/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp×
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMulÐ
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOpÀ
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMul_1¸
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/addÈ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOpÅ
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/BiasAdd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_30/split
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_1¡
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Reluµ
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_1ª
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/add_1
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_2
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Relu_1¹
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
ç^

(sequential_13_lstm_31_while_body_1674777H
Dsequential_13_lstm_31_while_sequential_13_lstm_31_while_loop_counterN
Jsequential_13_lstm_31_while_sequential_13_lstm_31_while_maximum_iterations+
'sequential_13_lstm_31_while_placeholder-
)sequential_13_lstm_31_while_placeholder_1-
)sequential_13_lstm_31_while_placeholder_2-
)sequential_13_lstm_31_while_placeholder_3G
Csequential_13_lstm_31_while_sequential_13_lstm_31_strided_slice_1_0
sequential_13_lstm_31_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_31_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_13_lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0:
´Ð_
Ksequential_13_lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0:
´ÐY
Jsequential_13_lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0:	Ð(
$sequential_13_lstm_31_while_identity*
&sequential_13_lstm_31_while_identity_1*
&sequential_13_lstm_31_while_identity_2*
&sequential_13_lstm_31_while_identity_3*
&sequential_13_lstm_31_while_identity_4*
&sequential_13_lstm_31_while_identity_5E
Asequential_13_lstm_31_while_sequential_13_lstm_31_strided_slice_1
}sequential_13_lstm_31_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_31_tensorarrayunstack_tensorlistfromtensor[
Gsequential_13_lstm_31_while_lstm_cell_31_matmul_readvariableop_resource:
´Ð]
Isequential_13_lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐW
Hsequential_13_lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource:	Ð¢?sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp¢>sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp¢@sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpï
Msequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2O
Msequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shapeØ
?sequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_13_lstm_31_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_31_tensorarrayunstack_tensorlistfromtensor_0'sequential_13_lstm_31_while_placeholderVsequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02A
?sequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItem
>sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOpIsequential_13_lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02@
>sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp¯
/sequential_13/lstm_31/while/lstm_cell_31/MatMulMatMulFsequential_13/lstm_31/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ21
/sequential_13/lstm_31/while/lstm_cell_31/MatMul
@sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOpKsequential_13_lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02B
@sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp
1sequential_13/lstm_31/while/lstm_cell_31/MatMul_1MatMul)sequential_13_lstm_31_while_placeholder_2Hsequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ23
1sequential_13/lstm_31/while/lstm_cell_31/MatMul_1
,sequential_13/lstm_31/while/lstm_cell_31/addAddV29sequential_13/lstm_31/while/lstm_cell_31/MatMul:product:0;sequential_13/lstm_31/while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2.
,sequential_13/lstm_31/while/lstm_cell_31/add
?sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOpJsequential_13_lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02A
?sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp
0sequential_13/lstm_31/while/lstm_cell_31/BiasAddBiasAdd0sequential_13/lstm_31/while/lstm_cell_31/add:z:0Gsequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ22
0sequential_13/lstm_31/while/lstm_cell_31/BiasAdd¶
8sequential_13/lstm_31/while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_13/lstm_31/while/lstm_cell_31/split/split_dimç
.sequential_13/lstm_31/while/lstm_cell_31/splitSplitAsequential_13/lstm_31/while/lstm_cell_31/split/split_dim:output:09sequential_13/lstm_31/while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split20
.sequential_13/lstm_31/while/lstm_cell_31/splitÛ
0sequential_13/lstm_31/while/lstm_cell_31/SigmoidSigmoid7sequential_13/lstm_31/while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´22
0sequential_13/lstm_31/while/lstm_cell_31/Sigmoidß
2sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_1Sigmoid7sequential_13/lstm_31/while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´24
2sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_1ù
,sequential_13/lstm_31/while/lstm_cell_31/mulMul6sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_1:y:0)sequential_13_lstm_31_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2.
,sequential_13/lstm_31/while/lstm_cell_31/mulÒ
-sequential_13/lstm_31/while/lstm_cell_31/ReluRelu7sequential_13/lstm_31/while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2/
-sequential_13/lstm_31/while/lstm_cell_31/Relu
.sequential_13/lstm_31/while/lstm_cell_31/mul_1Mul4sequential_13/lstm_31/while/lstm_cell_31/Sigmoid:y:0;sequential_13/lstm_31/while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´20
.sequential_13/lstm_31/while/lstm_cell_31/mul_1
.sequential_13/lstm_31/while/lstm_cell_31/add_1AddV20sequential_13/lstm_31/while/lstm_cell_31/mul:z:02sequential_13/lstm_31/while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´20
.sequential_13/lstm_31/while/lstm_cell_31/add_1ß
2sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_2Sigmoid7sequential_13/lstm_31/while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´24
2sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_2Ñ
/sequential_13/lstm_31/while/lstm_cell_31/Relu_1Relu2sequential_13/lstm_31/while/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´21
/sequential_13/lstm_31/while/lstm_cell_31/Relu_1
.sequential_13/lstm_31/while/lstm_cell_31/mul_2Mul6sequential_13/lstm_31/while/lstm_cell_31/Sigmoid_2:y:0=sequential_13/lstm_31/while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´20
.sequential_13/lstm_31/while/lstm_cell_31/mul_2Î
@sequential_13/lstm_31/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_13_lstm_31_while_placeholder_1'sequential_13_lstm_31_while_placeholder2sequential_13/lstm_31/while/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_13/lstm_31/while/TensorArrayV2Write/TensorListSetItem
!sequential_13/lstm_31/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_13/lstm_31/while/add/yÁ
sequential_13/lstm_31/while/addAddV2'sequential_13_lstm_31_while_placeholder*sequential_13/lstm_31/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_31/while/add
#sequential_13/lstm_31/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_13/lstm_31/while/add_1/yä
!sequential_13/lstm_31/while/add_1AddV2Dsequential_13_lstm_31_while_sequential_13_lstm_31_while_loop_counter,sequential_13/lstm_31/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_31/while/add_1Ã
$sequential_13/lstm_31/while/IdentityIdentity%sequential_13/lstm_31/while/add_1:z:0!^sequential_13/lstm_31/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_13/lstm_31/while/Identityì
&sequential_13/lstm_31/while/Identity_1IdentityJsequential_13_lstm_31_while_sequential_13_lstm_31_while_maximum_iterations!^sequential_13/lstm_31/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_31/while/Identity_1Å
&sequential_13/lstm_31/while/Identity_2Identity#sequential_13/lstm_31/while/add:z:0!^sequential_13/lstm_31/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_31/while/Identity_2ò
&sequential_13/lstm_31/while/Identity_3IdentityPsequential_13/lstm_31/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_13/lstm_31/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_31/while/Identity_3æ
&sequential_13/lstm_31/while/Identity_4Identity2sequential_13/lstm_31/while/lstm_cell_31/mul_2:z:0!^sequential_13/lstm_31/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2(
&sequential_13/lstm_31/while/Identity_4æ
&sequential_13/lstm_31/while/Identity_5Identity2sequential_13/lstm_31/while/lstm_cell_31/add_1:z:0!^sequential_13/lstm_31/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2(
&sequential_13/lstm_31/while/Identity_5Ì
 sequential_13/lstm_31/while/NoOpNoOp@^sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp?^sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOpA^sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_13/lstm_31/while/NoOp"U
$sequential_13_lstm_31_while_identity-sequential_13/lstm_31/while/Identity:output:0"Y
&sequential_13_lstm_31_while_identity_1/sequential_13/lstm_31/while/Identity_1:output:0"Y
&sequential_13_lstm_31_while_identity_2/sequential_13/lstm_31/while/Identity_2:output:0"Y
&sequential_13_lstm_31_while_identity_3/sequential_13/lstm_31/while/Identity_3:output:0"Y
&sequential_13_lstm_31_while_identity_4/sequential_13/lstm_31/while/Identity_4:output:0"Y
&sequential_13_lstm_31_while_identity_5/sequential_13/lstm_31/while/Identity_5:output:0"
Hsequential_13_lstm_31_while_lstm_cell_31_biasadd_readvariableop_resourceJsequential_13_lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0"
Isequential_13_lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resourceKsequential_13_lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0"
Gsequential_13_lstm_31_while_lstm_cell_31_matmul_readvariableop_resourceIsequential_13_lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0"
Asequential_13_lstm_31_while_sequential_13_lstm_31_strided_slice_1Csequential_13_lstm_31_while_sequential_13_lstm_31_strided_slice_1_0"
}sequential_13_lstm_31_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_31_tensorarrayunstack_tensorlistfromtensorsequential_13_lstm_31_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_31_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2
?sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp?sequential_13/lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp2
>sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp>sequential_13/lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp2
@sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp@sequential_13/lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 

e
G__inference_dropout_50_layer_call_and_return_conditional_losses_1681184

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs


I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1675581

inputs

states
states_12
matmul_readvariableop_resource:
´Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates
È
ù
.__inference_lstm_cell_31_layer_call_fn_1681465

inputs
states_0
states_1
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity

identity_1

identity_2¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_16763252
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
Ï

è
lstm_30_while_cond_1678899,
(lstm_30_while_lstm_30_while_loop_counter2
.lstm_30_while_lstm_30_while_maximum_iterations
lstm_30_while_placeholder
lstm_30_while_placeholder_1
lstm_30_while_placeholder_2
lstm_30_while_placeholder_3.
*lstm_30_while_less_lstm_30_strided_slice_1E
Alstm_30_while_lstm_30_while_cond_1678899___redundant_placeholder0E
Alstm_30_while_lstm_30_while_cond_1678899___redundant_placeholder1E
Alstm_30_while_lstm_30_while_cond_1678899___redundant_placeholder2E
Alstm_30_while_lstm_30_while_cond_1678899___redundant_placeholder3
lstm_30_while_identity

lstm_30/while/LessLesslstm_30_while_placeholder*lstm_30_while_less_lstm_30_strided_slice_1*
T0*
_output_shapes
: 2
lstm_30/while/Lessu
lstm_30/while/IdentityIdentitylstm_30/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_30/while/Identity"9
lstm_30_while_identitylstm_30/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
Ö
H
,__inference_dropout_48_layer_call_fn_1680464

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_16770292
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs


I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1675129

inputs

states
states_11
matmul_readvariableop_resource:	Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates
Þ
È
while_cond_1674996
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1674996___redundant_placeholder05
1while_while_cond_1674996___redundant_placeholder15
1while_while_cond_1674996___redundant_placeholder25
1while_while_cond_1674996___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
¯?
Ó
while_body_1677812
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	ÐI
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	ÐG
3while_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_29_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_29/BiasAdd/ReadVariableOp¢(while/lstm_cell_29/MatMul/ReadVariableOp¢*while/lstm_cell_29/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	Ð*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp×
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMulÐ
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOpÀ
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMul_1¸
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/addÈ
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOpÅ
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/BiasAdd
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_29/split
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_1¡
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Reluµ
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_1ª
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/add_1
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_2
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Relu_1¹
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
÷
ß
/__inference_sequential_13_layer_call_fn_1678216

inputs
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
	unknown_2:
´Ð
	unknown_3:
´Ð
	unknown_4:	Ð
	unknown_5:
´Ð
	unknown_6:
´Ð
	unknown_7:	Ð
	unknown_8:
´´
	unknown_9:	´

unknown_10:	´

unknown_11:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_16779772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
üU

D__inference_lstm_29_layer_call_and_return_conditional_losses_1679530
inputs_0>
+lstm_cell_29_matmul_readvariableop_resource:	ÐA
-lstm_cell_29_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_29_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_29/BiasAdd/ReadVariableOp¢"lstm_cell_29/MatMul/ReadVariableOp¢$lstm_cell_29/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp­
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul¼
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp©
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul_1 
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/add´
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp­
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim÷
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_29/split
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_1
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_1
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/add_1
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu_1¡
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1679446*
condR
while_cond_1679445*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ì 
ý
E__inference_dense_32_layer_call_and_return_conditional_losses_1681235

inputs4
!tensordot_readvariableop_resource:	´-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	´*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
³?
Õ
while_body_1681018
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_31_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_31/BiasAdd/ReadVariableOp¢(while/lstm_cell_31/MatMul/ReadVariableOp¢*while/lstm_cell_31/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp×
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMulÐ
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOpÀ
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMul_1¸
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/addÈ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOpÅ
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/BiasAdd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_31/split
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_1¡
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Reluµ
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_1ª
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/add_1
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_2
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Relu_1¹
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
ô%
ì
while_body_1674997
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_29_1675021_0:	Ð0
while_lstm_cell_29_1675023_0:
´Ð+
while_lstm_cell_29_1675025_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_29_1675021:	Ð.
while_lstm_cell_29_1675023:
´Ð)
while_lstm_cell_29_1675025:	Ð¢*while/lstm_cell_29/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemè
*while/lstm_cell_29/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_29_1675021_0while_lstm_cell_29_1675023_0while_lstm_cell_29_1675025_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_16749832,
*while/lstm_cell_29/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_29/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_29/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_29/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_29_1675021while_lstm_cell_29_1675021_0":
while_lstm_cell_29_1675023while_lstm_cell_29_1675023_0":
while_lstm_cell_29_1675025while_lstm_cell_29_1675025_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2X
*while/lstm_cell_29/StatefulPartitionedCall*while/lstm_cell_29/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
Ö
f
G__inference_dropout_47_layer_call_and_return_conditional_losses_1679843

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
·
¸
)__inference_lstm_30_layer_call_fn_1679887

inputs
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_16777082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs


*__inference_dense_31_layer_call_fn_1681138

inputs
unknown:
´´
	unknown_0:	´
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_16772192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Þ
È
while_cond_1676931
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1676931___redundant_placeholder05
1while_while_cond_1676931___redundant_placeholder15
1while_while_cond_1676931___redundant_placeholder25
1while_while_cond_1676931___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
ÿ?

D__inference_lstm_30_layer_call_and_return_conditional_losses_1675664

inputs(
lstm_cell_30_1675582:
´Ð(
lstm_cell_30_1675584:
´Ð#
lstm_cell_30_1675586:	Ð
identity¢$lstm_cell_30/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¤
$lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_30_1675582lstm_cell_30_1675584lstm_cell_30_1675586*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_16755812&
$lstm_cell_30/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÉ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_30_1675582lstm_cell_30_1675584lstm_cell_30_1675586*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1675595*
condR
while_cond_1675594*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identity}
NoOpNoOp%^lstm_cell_30/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 2L
$lstm_cell_30/StatefulPartitionedCall$lstm_cell_30/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Ï

è
lstm_29_while_cond_1678274,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3.
*lstm_29_while_less_lstm_29_strided_slice_1E
Alstm_29_while_lstm_29_while_cond_1678274___redundant_placeholder0E
Alstm_29_while_lstm_29_while_cond_1678274___redundant_placeholder1E
Alstm_29_while_lstm_29_while_cond_1678274___redundant_placeholder2E
Alstm_29_while_lstm_29_while_cond_1678274___redundant_placeholder3
lstm_29_while_identity

lstm_29/while/LessLesslstm_29_while_placeholder*lstm_29_while_less_lstm_29_strided_slice_1*
T0*
_output_shapes
: 2
lstm_29/while/Lessu
lstm_29/while/IdentityIdentitylstm_29/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_29/while/Identity"9
lstm_29_while_identitylstm_29/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
ÿ?

D__inference_lstm_30_layer_call_and_return_conditional_losses_1675866

inputs(
lstm_cell_30_1675784:
´Ð(
lstm_cell_30_1675786:
´Ð#
lstm_cell_30_1675788:	Ð
identity¢$lstm_cell_30/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¤
$lstm_cell_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_30_1675784lstm_cell_30_1675786lstm_cell_30_1675788*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_16757272&
$lstm_cell_30/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÉ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_30_1675784lstm_cell_30_1675786lstm_cell_30_1675788*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1675797*
condR
while_cond_1675796*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identity}
NoOpNoOp%^lstm_cell_30/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 2L
$lstm_cell_30/StatefulPartitionedCall$lstm_cell_30/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
¯?
Ó
while_body_1679732
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	ÐI
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	ÐG
3while_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_29_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_29/BiasAdd/ReadVariableOp¢(while/lstm_cell_29/MatMul/ReadVariableOp¢*while/lstm_cell_29/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	Ð*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp×
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMulÐ
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOpÀ
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMul_1¸
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/addÈ
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOpÅ
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/BiasAdd
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_29/split
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_1¡
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Reluµ
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_1ª
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/add_1
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_2
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Relu_1¹
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
Ö
f
G__inference_dropout_47_layer_call_and_return_conditional_losses_1677737

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs


I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1676325

inputs

states
states_12
matmul_readvariableop_resource:
´Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates


I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1681529

inputs
states_0
states_12
matmul_readvariableop_resource:
´Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
¯?
Ó
while_body_1676775
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_29_matmul_readvariableop_resource_0:	ÐI
5while_lstm_cell_29_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_29_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_29_matmul_readvariableop_resource:	ÐG
3while_lstm_cell_29_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_29_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_29/BiasAdd/ReadVariableOp¢(while/lstm_cell_29/MatMul/ReadVariableOp¢*while/lstm_cell_29/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_29/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_29_matmul_readvariableop_resource_0*
_output_shapes
:	Ð*
dtype02*
(while/lstm_cell_29/MatMul/ReadVariableOp×
while/lstm_cell_29/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMulÐ
*while/lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_29_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_29/MatMul_1/ReadVariableOpÀ
while/lstm_cell_29/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/MatMul_1¸
while/lstm_cell_29/addAddV2#while/lstm_cell_29/MatMul:product:0%while/lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/addÈ
)while/lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_29_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_29/BiasAdd/ReadVariableOpÅ
while/lstm_cell_29/BiasAddBiasAddwhile/lstm_cell_29/add:z:01while/lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_29/BiasAdd
"while/lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_29/split/split_dim
while/lstm_cell_29/splitSplit+while/lstm_cell_29/split/split_dim:output:0#while/lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_29/split
while/lstm_cell_29/SigmoidSigmoid!while/lstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid
while/lstm_cell_29/Sigmoid_1Sigmoid!while/lstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_1¡
while/lstm_cell_29/mulMul while/lstm_cell_29/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul
while/lstm_cell_29/ReluRelu!while/lstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Reluµ
while/lstm_cell_29/mul_1Mulwhile/lstm_cell_29/Sigmoid:y:0%while/lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_1ª
while/lstm_cell_29/add_1AddV2while/lstm_cell_29/mul:z:0while/lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/add_1
while/lstm_cell_29/Sigmoid_2Sigmoid!while/lstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Sigmoid_2
while/lstm_cell_29/Relu_1Reluwhile/lstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/Relu_1¹
while/lstm_cell_29/mul_2Mul while/lstm_cell_29/Sigmoid_2:y:0'while/lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_29/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_29/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_29/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_29/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_29/BiasAdd/ReadVariableOp)^while/lstm_cell_29/MatMul/ReadVariableOp+^while/lstm_cell_29/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_29_biasadd_readvariableop_resource4while_lstm_cell_29_biasadd_readvariableop_resource_0"l
3while_lstm_cell_29_matmul_1_readvariableop_resource5while_lstm_cell_29_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_29_matmul_readvariableop_resource3while_lstm_cell_29_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_29/BiasAdd/ReadVariableOp)while/lstm_cell_29/BiasAdd/ReadVariableOp2T
(while/lstm_cell_29/MatMul/ReadVariableOp(while/lstm_cell_29/MatMul/ReadVariableOp2X
*while/lstm_cell_29/MatMul_1/ReadVariableOp*while/lstm_cell_29/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
Ö
H
,__inference_dropout_49_layer_call_fn_1681107

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_16771862
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs

æ
/__inference_sequential_13_layer_call_fn_1678037
lstm_29_input
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
	unknown_2:
´Ð
	unknown_3:
´Ð
	unknown_4:	Ð
	unknown_5:
´Ð
	unknown_6:
´Ð
	unknown_7:	Ð
	unknown_8:
´´
	unknown_9:	´

unknown_10:	´

unknown_11:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_29_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_16779772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_29_input
V
 
D__inference_lstm_31_layer_call_and_return_conditional_losses_1680673
inputs_0?
+lstm_cell_31_matmul_readvariableop_resource:
´ÐA
-lstm_cell_31_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_31_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_31/BiasAdd/ReadVariableOp¢"lstm_cell_31/MatMul/ReadVariableOp¢$lstm_cell_31/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp­
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul¼
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp©
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul_1 
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/add´
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp­
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim÷
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_31/split
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_1
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_1
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/add_1
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu_1¡
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1680589*
condR
while_cond_1680588*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
inputs/0
Ö
f
G__inference_dropout_49_layer_call_and_return_conditional_losses_1681129

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
ÃU

D__inference_lstm_30_layer_call_and_return_conditional_losses_1680316

inputs?
+lstm_cell_30_matmul_readvariableop_resource:
´ÐA
-lstm_cell_30_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_30_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_30/BiasAdd/ReadVariableOp¢"lstm_cell_30/MatMul/ReadVariableOp¢$lstm_cell_30/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp­
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul¼
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp©
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul_1 
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/add´
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp­
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim÷
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_30/split
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_1
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_1
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/add_1
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu_1¡
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1680232*
condR
while_cond_1680231*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
á
º
)__inference_lstm_30_layer_call_fn_1679865
inputs_0
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_16758662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
inputs/0
á
º
)__inference_lstm_31_layer_call_fn_1680508
inputs_0
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_16764642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
inputs/0
´
·
)__inference_lstm_29_layer_call_fn_1679233

inputs
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_16768592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö!
ÿ
E__inference_dense_31_layer_call_and_return_conditional_losses_1681169

inputs5
!tensordot_readvariableop_resource:
´´.
biasadd_readvariableop_resource:	´
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
´´*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:´2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:´*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
V
 
D__inference_lstm_30_layer_call_and_return_conditional_losses_1680030
inputs_0?
+lstm_cell_30_matmul_readvariableop_resource:
´ÐA
-lstm_cell_30_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_30_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_30/BiasAdd/ReadVariableOp¢"lstm_cell_30/MatMul/ReadVariableOp¢$lstm_cell_30/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp­
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul¼
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp©
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul_1 
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/add´
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp­
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim÷
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_30/split
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_1
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_1
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/add_1
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu_1¡
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1679946*
condR
while_cond_1679945*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
inputs/0
±1
ò
J__inference_sequential_13_layer_call_and_return_conditional_losses_1677977

inputs"
lstm_29_1677941:	Ð#
lstm_29_1677943:
´Ð
lstm_29_1677945:	Ð#
lstm_30_1677949:
´Ð#
lstm_30_1677951:
´Ð
lstm_30_1677953:	Ð#
lstm_31_1677957:
´Ð#
lstm_31_1677959:
´Ð
lstm_31_1677961:	Ð$
dense_31_1677965:
´´
dense_31_1677967:	´#
dense_32_1677971:	´
dense_32_1677973:
identity¢ dense_31/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢"dropout_47/StatefulPartitionedCall¢"dropout_48/StatefulPartitionedCall¢"dropout_49/StatefulPartitionedCall¢"dropout_50/StatefulPartitionedCall¢lstm_29/StatefulPartitionedCall¢lstm_30/StatefulPartitionedCall¢lstm_31/StatefulPartitionedCallª
lstm_29/StatefulPartitionedCallStatefulPartitionedCallinputslstm_29_1677941lstm_29_1677943lstm_29_1677945*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_16778962!
lstm_29/StatefulPartitionedCall
"dropout_47/StatefulPartitionedCallStatefulPartitionedCall(lstm_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_16777372$
"dropout_47/StatefulPartitionedCallÏ
lstm_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_47/StatefulPartitionedCall:output:0lstm_30_1677949lstm_30_1677951lstm_30_1677953*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_16777082!
lstm_30/StatefulPartitionedCall¿
"dropout_48/StatefulPartitionedCallStatefulPartitionedCall(lstm_30/StatefulPartitionedCall:output:0#^dropout_47/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_16775492$
"dropout_48/StatefulPartitionedCallÏ
lstm_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_48/StatefulPartitionedCall:output:0lstm_31_1677957lstm_31_1677959lstm_31_1677961*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_16775202!
lstm_31/StatefulPartitionedCall¿
"dropout_49/StatefulPartitionedCallStatefulPartitionedCall(lstm_31/StatefulPartitionedCall:output:0#^dropout_48/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_16773612$
"dropout_49/StatefulPartitionedCallÁ
 dense_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_49/StatefulPartitionedCall:output:0dense_31_1677965dense_31_1677967*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_16772192"
 dense_31/StatefulPartitionedCallÀ
"dropout_50/StatefulPartitionedCallStatefulPartitionedCall)dense_31/StatefulPartitionedCall:output:0#^dropout_49/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_16773282$
"dropout_50/StatefulPartitionedCallÀ
 dense_32/StatefulPartitionedCallStatefulPartitionedCall+dropout_50/StatefulPartitionedCall:output:0dense_32_1677971dense_32_1677973*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_16772622"
 dense_32/StatefulPartitionedCall
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall#^dropout_47/StatefulPartitionedCall#^dropout_48/StatefulPartitionedCall#^dropout_49/StatefulPartitionedCall#^dropout_50/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall ^lstm_30/StatefulPartitionedCall ^lstm_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2H
"dropout_47/StatefulPartitionedCall"dropout_47/StatefulPartitionedCall2H
"dropout_48/StatefulPartitionedCall"dropout_48/StatefulPartitionedCall2H
"dropout_49/StatefulPartitionedCall"dropout_49/StatefulPartitionedCall2H
"dropout_50/StatefulPartitionedCall"dropout_50/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall2B
lstm_30/StatefulPartitionedCalllstm_30/StatefulPartitionedCall2B
lstm_31/StatefulPartitionedCalllstm_31/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÃU

D__inference_lstm_31_layer_call_and_return_conditional_losses_1680959

inputs?
+lstm_cell_31_matmul_readvariableop_resource:
´ÐA
-lstm_cell_31_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_31_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_31/BiasAdd/ReadVariableOp¢"lstm_cell_31/MatMul/ReadVariableOp¢$lstm_cell_31/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_31/MatMul/ReadVariableOpReadVariableOp+lstm_cell_31_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_31/MatMul/ReadVariableOp­
lstm_cell_31/MatMulMatMulstrided_slice_2:output:0*lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul¼
$lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_31_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_31/MatMul_1/ReadVariableOp©
lstm_cell_31/MatMul_1MatMulzeros:output:0,lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/MatMul_1 
lstm_cell_31/addAddV2lstm_cell_31/MatMul:product:0lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/add´
#lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_31_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_31/BiasAdd/ReadVariableOp­
lstm_cell_31/BiasAddBiasAddlstm_cell_31/add:z:0+lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_31/BiasAdd~
lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_31/split/split_dim÷
lstm_cell_31/splitSplit%lstm_cell_31/split/split_dim:output:0lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_31/split
lstm_cell_31/SigmoidSigmoidlstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid
lstm_cell_31/Sigmoid_1Sigmoidlstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_1
lstm_cell_31/mulMullstm_cell_31/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul~
lstm_cell_31/ReluRelulstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu
lstm_cell_31/mul_1Mullstm_cell_31/Sigmoid:y:0lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_1
lstm_cell_31/add_1AddV2lstm_cell_31/mul:z:0lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/add_1
lstm_cell_31/Sigmoid_2Sigmoidlstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Sigmoid_2}
lstm_cell_31/Relu_1Relulstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/Relu_1¡
lstm_cell_31/mul_2Mullstm_cell_31/Sigmoid_2:y:0!lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_31/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_31_matmul_readvariableop_resource-lstm_cell_31_matmul_1_readvariableop_resource,lstm_cell_31_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1680875*
condR
while_cond_1680874*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_31/BiasAdd/ReadVariableOp#^lstm_cell_31/MatMul/ReadVariableOp%^lstm_cell_31/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_31/BiasAdd/ReadVariableOp#lstm_cell_31/BiasAdd/ReadVariableOp2H
"lstm_cell_31/MatMul/ReadVariableOp"lstm_cell_31/MatMul/ReadVariableOp2L
$lstm_cell_31/MatMul_1/ReadVariableOp$lstm_cell_31/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
½U

D__inference_lstm_29_layer_call_and_return_conditional_losses_1677896

inputs>
+lstm_cell_29_matmul_readvariableop_resource:	ÐA
-lstm_cell_29_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_29_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_29/BiasAdd/ReadVariableOp¢"lstm_cell_29/MatMul/ReadVariableOp¢$lstm_cell_29/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_29/MatMul/ReadVariableOpReadVariableOp+lstm_cell_29_matmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02$
"lstm_cell_29/MatMul/ReadVariableOp­
lstm_cell_29/MatMulMatMulstrided_slice_2:output:0*lstm_cell_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul¼
$lstm_cell_29/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_29_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_29/MatMul_1/ReadVariableOp©
lstm_cell_29/MatMul_1MatMulzeros:output:0,lstm_cell_29/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/MatMul_1 
lstm_cell_29/addAddV2lstm_cell_29/MatMul:product:0lstm_cell_29/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/add´
#lstm_cell_29/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_29_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_29/BiasAdd/ReadVariableOp­
lstm_cell_29/BiasAddBiasAddlstm_cell_29/add:z:0+lstm_cell_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_29/BiasAdd~
lstm_cell_29/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_29/split/split_dim÷
lstm_cell_29/splitSplit%lstm_cell_29/split/split_dim:output:0lstm_cell_29/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_29/split
lstm_cell_29/SigmoidSigmoidlstm_cell_29/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid
lstm_cell_29/Sigmoid_1Sigmoidlstm_cell_29/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_1
lstm_cell_29/mulMullstm_cell_29/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul~
lstm_cell_29/ReluRelulstm_cell_29/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu
lstm_cell_29/mul_1Mullstm_cell_29/Sigmoid:y:0lstm_cell_29/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_1
lstm_cell_29/add_1AddV2lstm_cell_29/mul:z:0lstm_cell_29/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/add_1
lstm_cell_29/Sigmoid_2Sigmoidlstm_cell_29/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Sigmoid_2}
lstm_cell_29/Relu_1Relulstm_cell_29/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/Relu_1¡
lstm_cell_29/mul_2Mullstm_cell_29/Sigmoid_2:y:0!lstm_cell_29/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_29/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_29_matmul_readvariableop_resource-lstm_cell_29_matmul_1_readvariableop_resource,lstm_cell_29_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1677812*
condR
while_cond_1677811*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_29/BiasAdd/ReadVariableOp#^lstm_cell_29/MatMul/ReadVariableOp%^lstm_cell_29/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_29/BiasAdd/ReadVariableOp#lstm_cell_29/BiasAdd/ReadVariableOp2H
"lstm_cell_29/MatMul/ReadVariableOp"lstm_cell_29/MatMul/ReadVariableOp2L
$lstm_cell_29/MatMul_1/ReadVariableOp$lstm_cell_29/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

è
lstm_30_while_cond_1678414,
(lstm_30_while_lstm_30_while_loop_counter2
.lstm_30_while_lstm_30_while_maximum_iterations
lstm_30_while_placeholder
lstm_30_while_placeholder_1
lstm_30_while_placeholder_2
lstm_30_while_placeholder_3.
*lstm_30_while_less_lstm_30_strided_slice_1E
Alstm_30_while_lstm_30_while_cond_1678414___redundant_placeholder0E
Alstm_30_while_lstm_30_while_cond_1678414___redundant_placeholder1E
Alstm_30_while_lstm_30_while_cond_1678414___redundant_placeholder2E
Alstm_30_while_lstm_30_while_cond_1678414___redundant_placeholder3
lstm_30_while_identity

lstm_30/while/LessLesslstm_30_while_placeholder*lstm_30_while_less_lstm_30_strided_slice_1*
T0*
_output_shapes
: 2
lstm_30/while/Lessu
lstm_30/while/IdentityIdentitylstm_30/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_30/while/Identity"9
lstm_30_while_identitylstm_30/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
È
ù
.__inference_lstm_cell_31_layer_call_fn_1681448

inputs
states_0
states_1
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity

identity_1

identity_2¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_16761792
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
Ö
H
,__inference_dropout_50_layer_call_fn_1681174

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_16772302
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
á
º
)__inference_lstm_30_layer_call_fn_1679854
inputs_0
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_16756642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
inputs/0


I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1681301

inputs
states_0
states_11
matmul_readvariableop_resource:	Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
³?
Õ
while_body_1677089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_31_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_31_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_31_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_31_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_31_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_31/BiasAdd/ReadVariableOp¢(while/lstm_cell_31/MatMul/ReadVariableOp¢*while/lstm_cell_31/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_31/MatMul/ReadVariableOp×
while/lstm_cell_31/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMulÐ
*while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_31/MatMul_1/ReadVariableOpÀ
while/lstm_cell_31/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/MatMul_1¸
while/lstm_cell_31/addAddV2#while/lstm_cell_31/MatMul:product:0%while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/addÈ
)while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_31/BiasAdd/ReadVariableOpÅ
while/lstm_cell_31/BiasAddBiasAddwhile/lstm_cell_31/add:z:01while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_31/BiasAdd
"while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_31/split/split_dim
while/lstm_cell_31/splitSplit+while/lstm_cell_31/split/split_dim:output:0#while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_31/split
while/lstm_cell_31/SigmoidSigmoid!while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid
while/lstm_cell_31/Sigmoid_1Sigmoid!while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_1¡
while/lstm_cell_31/mulMul while/lstm_cell_31/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul
while/lstm_cell_31/ReluRelu!while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Reluµ
while/lstm_cell_31/mul_1Mulwhile/lstm_cell_31/Sigmoid:y:0%while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_1ª
while/lstm_cell_31/add_1AddV2while/lstm_cell_31/mul:z:0while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/add_1
while/lstm_cell_31/Sigmoid_2Sigmoid!while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Sigmoid_2
while/lstm_cell_31/Relu_1Reluwhile/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/Relu_1¹
while/lstm_cell_31/mul_2Mul while/lstm_cell_31/Sigmoid_2:y:0'while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_31/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_31/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_31/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_31/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_31/BiasAdd/ReadVariableOp)^while/lstm_cell_31/MatMul/ReadVariableOp+^while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_31_biasadd_readvariableop_resource4while_lstm_cell_31_biasadd_readvariableop_resource_0"l
3while_lstm_cell_31_matmul_1_readvariableop_resource5while_lstm_cell_31_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_31_matmul_readvariableop_resource3while_lstm_cell_31_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_31/BiasAdd/ReadVariableOp)while/lstm_cell_31/BiasAdd/ReadVariableOp2T
(while/lstm_cell_31/MatMul/ReadVariableOp(while/lstm_cell_31/MatMul/ReadVariableOp2X
*while/lstm_cell_31/MatMul_1/ReadVariableOp*while/lstm_cell_31/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1677811
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1677811___redundant_placeholder05
1while_while_cond_1677811___redundant_placeholder15
1while_while_cond_1677811___redundant_placeholder25
1while_while_cond_1677811___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
Ö
H
,__inference_dropout_47_layer_call_fn_1679821

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_16768722
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
³?
Õ
while_body_1680375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_30_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_30/BiasAdd/ReadVariableOp¢(while/lstm_cell_30/MatMul/ReadVariableOp¢*while/lstm_cell_30/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp×
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMulÐ
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOpÀ
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMul_1¸
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/addÈ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOpÅ
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/BiasAdd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_30/split
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_1¡
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Reluµ
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_1ª
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/add_1
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_2
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Relu_1¹
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 


I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1681431

inputs
states_0
states_12
matmul_readvariableop_resource:
´Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
Þ
È
while_cond_1679588
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1679588___redundant_placeholder05
1while_while_cond_1679588___redundant_placeholder15
1while_while_cond_1679588___redundant_placeholder25
1while_while_cond_1679588___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
Ú
Ü
%__inference_signature_wrapper_1678154
lstm_29_input
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
	unknown_2:
´Ð
	unknown_3:
´Ð
	unknown_4:	Ð
	unknown_5:
´Ð
	unknown_6:
´Ð
	unknown_7:	Ð
	unknown_8:
´´
	unknown_9:	´

unknown_10:	´

unknown_11:
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCalllstm_29_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_16749162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_29_input
Þ
È
while_cond_1680231
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1680231___redundant_placeholder05
1while_while_cond_1680231___redundant_placeholder15
1while_while_cond_1680231___redundant_placeholder25
1while_while_cond_1680231___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:

e
G__inference_dropout_49_layer_call_and_return_conditional_losses_1681117

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs


I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1675727

inputs

states
states_12
matmul_readvariableop_resource:
´Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_namestates
èJ
Õ

lstm_30_while_body_1678415,
(lstm_30_while_lstm_30_while_loop_counter2
.lstm_30_while_lstm_30_while_maximum_iterations
lstm_30_while_placeholder
lstm_30_while_placeholder_1
lstm_30_while_placeholder_2
lstm_30_while_placeholder_3+
'lstm_30_while_lstm_30_strided_slice_1_0g
clstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0:
´ÐQ
=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0:
´ÐK
<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0:	Ð
lstm_30_while_identity
lstm_30_while_identity_1
lstm_30_while_identity_2
lstm_30_while_identity_3
lstm_30_while_identity_4
lstm_30_while_identity_5)
%lstm_30_while_lstm_30_strided_slice_1e
alstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensorM
9lstm_30_while_lstm_cell_30_matmul_readvariableop_resource:
´ÐO
;lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐI
:lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource:	Ð¢1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp¢0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp¢2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpÓ
?lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2A
?lstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_30/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0lstm_30_while_placeholderHlstm_30/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype023
1lstm_30/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype022
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp÷
!lstm_30/while/lstm_cell_30/MatMulMatMul8lstm_30/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2#
!lstm_30/while/lstm_cell_30/MatMulè
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype024
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOpà
#lstm_30/while/lstm_cell_30/MatMul_1MatMullstm_30_while_placeholder_2:lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2%
#lstm_30/while/lstm_cell_30/MatMul_1Ø
lstm_30/while/lstm_cell_30/addAddV2+lstm_30/while/lstm_cell_30/MatMul:product:0-lstm_30/while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2 
lstm_30/while/lstm_cell_30/addà
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype023
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOpå
"lstm_30/while/lstm_cell_30/BiasAddBiasAdd"lstm_30/while/lstm_cell_30/add:z:09lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2$
"lstm_30/while/lstm_cell_30/BiasAdd
*lstm_30/while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_30/while/lstm_cell_30/split/split_dim¯
 lstm_30/while/lstm_cell_30/splitSplit3lstm_30/while/lstm_cell_30/split/split_dim:output:0+lstm_30/while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2"
 lstm_30/while/lstm_cell_30/split±
"lstm_30/while/lstm_cell_30/SigmoidSigmoid)lstm_30/while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2$
"lstm_30/while/lstm_cell_30/Sigmoidµ
$lstm_30/while/lstm_cell_30/Sigmoid_1Sigmoid)lstm_30/while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_30/while/lstm_cell_30/Sigmoid_1Á
lstm_30/while/lstm_cell_30/mulMul(lstm_30/while/lstm_cell_30/Sigmoid_1:y:0lstm_30_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_30/while/lstm_cell_30/mul¨
lstm_30/while/lstm_cell_30/ReluRelu)lstm_30/while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
lstm_30/while/lstm_cell_30/ReluÕ
 lstm_30/while/lstm_cell_30/mul_1Mul&lstm_30/while/lstm_cell_30/Sigmoid:y:0-lstm_30/while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_30/while/lstm_cell_30/mul_1Ê
 lstm_30/while/lstm_cell_30/add_1AddV2"lstm_30/while/lstm_cell_30/mul:z:0$lstm_30/while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_30/while/lstm_cell_30/add_1µ
$lstm_30/while/lstm_cell_30/Sigmoid_2Sigmoid)lstm_30/while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_30/while/lstm_cell_30/Sigmoid_2§
!lstm_30/while/lstm_cell_30/Relu_1Relu$lstm_30/while/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!lstm_30/while/lstm_cell_30/Relu_1Ù
 lstm_30/while/lstm_cell_30/mul_2Mul(lstm_30/while/lstm_cell_30/Sigmoid_2:y:0/lstm_30/while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_30/while/lstm_cell_30/mul_2
2lstm_30/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_30_while_placeholder_1lstm_30_while_placeholder$lstm_30/while/lstm_cell_30/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_30/while/TensorArrayV2Write/TensorListSetIteml
lstm_30/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_30/while/add/y
lstm_30/while/addAddV2lstm_30_while_placeholderlstm_30/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_30/while/addp
lstm_30/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_30/while/add_1/y
lstm_30/while/add_1AddV2(lstm_30_while_lstm_30_while_loop_counterlstm_30/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_30/while/add_1
lstm_30/while/IdentityIdentitylstm_30/while/add_1:z:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity¦
lstm_30/while/Identity_1Identity.lstm_30_while_lstm_30_while_maximum_iterations^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_1
lstm_30/while/Identity_2Identitylstm_30/while/add:z:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_2º
lstm_30/while/Identity_3IdentityBlstm_30/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_30/while/NoOp*
T0*
_output_shapes
: 2
lstm_30/while/Identity_3®
lstm_30/while/Identity_4Identity$lstm_30/while/lstm_cell_30/mul_2:z:0^lstm_30/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/while/Identity_4®
lstm_30/while/Identity_5Identity$lstm_30/while/lstm_cell_30/add_1:z:0^lstm_30/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_30/while/Identity_5
lstm_30/while/NoOpNoOp2^lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp1^lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp3^lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_30/while/NoOp"9
lstm_30_while_identitylstm_30/while/Identity:output:0"=
lstm_30_while_identity_1!lstm_30/while/Identity_1:output:0"=
lstm_30_while_identity_2!lstm_30/while/Identity_2:output:0"=
lstm_30_while_identity_3!lstm_30/while/Identity_3:output:0"=
lstm_30_while_identity_4!lstm_30/while/Identity_4:output:0"=
lstm_30_while_identity_5!lstm_30/while/Identity_5:output:0"P
%lstm_30_while_lstm_30_strided_slice_1'lstm_30_while_lstm_30_strided_slice_1_0"z
:lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource<lstm_30_while_lstm_cell_30_biasadd_readvariableop_resource_0"|
;lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource=lstm_30_while_lstm_cell_30_matmul_1_readvariableop_resource_0"x
9lstm_30_while_lstm_cell_30_matmul_readvariableop_resource;lstm_30_while_lstm_cell_30_matmul_readvariableop_resource_0"È
alstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensorclstm_30_while_tensorarrayv2read_tensorlistgetitem_lstm_30_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2f
1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp1lstm_30/while/lstm_cell_30/BiasAdd/ReadVariableOp2d
0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp0lstm_30/while/lstm_cell_30/MatMul/ReadVariableOp2h
2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp2lstm_30/while/lstm_cell_30/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 


I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1681497

inputs
states_0
states_12
matmul_readvariableop_resource:
´Ð4
 matmul_1_readvariableop_resource:
´Ð.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
Ï

è
lstm_31_while_cond_1678554,
(lstm_31_while_lstm_31_while_loop_counter2
.lstm_31_while_lstm_31_while_maximum_iterations
lstm_31_while_placeholder
lstm_31_while_placeholder_1
lstm_31_while_placeholder_2
lstm_31_while_placeholder_3.
*lstm_31_while_less_lstm_31_strided_slice_1E
Alstm_31_while_lstm_31_while_cond_1678554___redundant_placeholder0E
Alstm_31_while_lstm_31_while_cond_1678554___redundant_placeholder1E
Alstm_31_while_lstm_31_while_cond_1678554___redundant_placeholder2E
Alstm_31_while_lstm_31_while_cond_1678554___redundant_placeholder3
lstm_31_while_identity

lstm_31/while/LessLesslstm_31_while_placeholder*lstm_31_while_less_lstm_31_strided_slice_1*
T0*
_output_shapes
: 2
lstm_31/while/Lessu
lstm_31/while/IdentityIdentitylstm_31/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_31/while/Identity"9
lstm_31_while_identitylstm_31/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
Ï

è
lstm_29_while_cond_1678752,
(lstm_29_while_lstm_29_while_loop_counter2
.lstm_29_while_lstm_29_while_maximum_iterations
lstm_29_while_placeholder
lstm_29_while_placeholder_1
lstm_29_while_placeholder_2
lstm_29_while_placeholder_3.
*lstm_29_while_less_lstm_29_strided_slice_1E
Alstm_29_while_lstm_29_while_cond_1678752___redundant_placeholder0E
Alstm_29_while_lstm_29_while_cond_1678752___redundant_placeholder1E
Alstm_29_while_lstm_29_while_cond_1678752___redundant_placeholder2E
Alstm_29_while_lstm_29_while_cond_1678752___redundant_placeholder3
lstm_29_while_identity

lstm_29/while/LessLesslstm_29_while_placeholder*lstm_29_while_less_lstm_29_strided_slice_1*
T0*
_output_shapes
: 2
lstm_29/while/Lessu
lstm_29/while/IdentityIdentitylstm_29/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_29/while/Identity"9
lstm_29_while_identitylstm_29/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
ò*
Þ
J__inference_sequential_13_layer_call_and_return_conditional_losses_1677269

inputs"
lstm_29_1676860:	Ð#
lstm_29_1676862:
´Ð
lstm_29_1676864:	Ð#
lstm_30_1677017:
´Ð#
lstm_30_1677019:
´Ð
lstm_30_1677021:	Ð#
lstm_31_1677174:
´Ð#
lstm_31_1677176:
´Ð
lstm_31_1677178:	Ð$
dense_31_1677220:
´´
dense_31_1677222:	´#
dense_32_1677263:	´
dense_32_1677265:
identity¢ dense_31/StatefulPartitionedCall¢ dense_32/StatefulPartitionedCall¢lstm_29/StatefulPartitionedCall¢lstm_30/StatefulPartitionedCall¢lstm_31/StatefulPartitionedCallª
lstm_29/StatefulPartitionedCallStatefulPartitionedCallinputslstm_29_1676860lstm_29_1676862lstm_29_1676864*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_29_layer_call_and_return_conditional_losses_16768592!
lstm_29/StatefulPartitionedCall
dropout_47/PartitionedCallPartitionedCall(lstm_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_47_layer_call_and_return_conditional_losses_16768722
dropout_47/PartitionedCallÇ
lstm_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_47/PartitionedCall:output:0lstm_30_1677017lstm_30_1677019lstm_30_1677021*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_30_layer_call_and_return_conditional_losses_16770162!
lstm_30/StatefulPartitionedCall
dropout_48/PartitionedCallPartitionedCall(lstm_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_48_layer_call_and_return_conditional_losses_16770292
dropout_48/PartitionedCallÇ
lstm_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_48/PartitionedCall:output:0lstm_31_1677174lstm_31_1677176lstm_31_1677178*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_31_layer_call_and_return_conditional_losses_16771732!
lstm_31/StatefulPartitionedCall
dropout_49/PartitionedCallPartitionedCall(lstm_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_49_layer_call_and_return_conditional_losses_16771862
dropout_49/PartitionedCall¹
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_49/PartitionedCall:output:0dense_31_1677220dense_31_1677222*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_16772192"
 dense_31/StatefulPartitionedCall
dropout_50/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_50_layer_call_and_return_conditional_losses_16772302
dropout_50/PartitionedCall¸
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#dropout_50/PartitionedCall:output:0dense_32_1677263dense_32_1677265*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_16772622"
 dense_32/StatefulPartitionedCall
IdentityIdentity)dense_32/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityú
NoOpNoOp!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall ^lstm_29/StatefulPartitionedCall ^lstm_30/StatefulPartitionedCall ^lstm_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2B
lstm_29/StatefulPartitionedCalllstm_29/StatefulPartitionedCall2B
lstm_30/StatefulPartitionedCalllstm_30/StatefulPartitionedCall2B
lstm_31/StatefulPartitionedCalllstm_31/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
f
G__inference_dropout_50_layer_call_and_return_conditional_losses_1681196

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
Å
ø
.__inference_lstm_cell_29_layer_call_fn_1681252

inputs
states_0
states_1
unknown:	Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity

identity_1

identity_2¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_16749832
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
Þ
È
while_cond_1675594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1675594___redundant_placeholder05
1while_while_cond_1675594___redundant_placeholder15
1while_while_cond_1675594___redundant_placeholder25
1while_while_cond_1675594___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
Þ
È
while_cond_1680731
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1680731___redundant_placeholder05
1while_while_cond_1680731___redundant_placeholder15
1while_while_cond_1680731___redundant_placeholder25
1while_while_cond_1680731___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
Þ
È
while_cond_1681017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1681017___redundant_placeholder05
1while_while_cond_1681017___redundant_placeholder15
1while_while_cond_1681017___redundant_placeholder25
1while_while_cond_1681017___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
:
ÃU

D__inference_lstm_30_layer_call_and_return_conditional_losses_1677708

inputs?
+lstm_cell_30_matmul_readvariableop_resource:
´ÐA
-lstm_cell_30_matmul_1_readvariableop_resource:
´Ð;
,lstm_cell_30_biasadd_readvariableop_resource:	Ð
identity¢#lstm_cell_30/BiasAdd/ReadVariableOp¢"lstm_cell_30/MatMul/ReadVariableOp¢$lstm_cell_30/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :´2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ´2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :´2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_30/MatMul/ReadVariableOpReadVariableOp+lstm_cell_30_matmul_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02$
"lstm_cell_30/MatMul/ReadVariableOp­
lstm_cell_30/MatMulMatMulstrided_slice_2:output:0*lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul¼
$lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_30_matmul_1_readvariableop_resource* 
_output_shapes
:
´Ð*
dtype02&
$lstm_cell_30/MatMul_1/ReadVariableOp©
lstm_cell_30/MatMul_1MatMulzeros:output:0,lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/MatMul_1 
lstm_cell_30/addAddV2lstm_cell_30/MatMul:product:0lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/add´
#lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_30_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_30/BiasAdd/ReadVariableOp­
lstm_cell_30/BiasAddBiasAddlstm_cell_30/add:z:0+lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_30/BiasAdd~
lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_30/split/split_dim÷
lstm_cell_30/splitSplit%lstm_cell_30/split/split_dim:output:0lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
lstm_cell_30/split
lstm_cell_30/SigmoidSigmoidlstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid
lstm_cell_30/Sigmoid_1Sigmoidlstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_1
lstm_cell_30/mulMullstm_cell_30/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul~
lstm_cell_30/ReluRelulstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu
lstm_cell_30/mul_1Mullstm_cell_30/Sigmoid:y:0lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_1
lstm_cell_30/add_1AddV2lstm_cell_30/mul:z:0lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/add_1
lstm_cell_30/Sigmoid_2Sigmoidlstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Sigmoid_2}
lstm_cell_30/Relu_1Relulstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/Relu_1¡
lstm_cell_30/mul_2Mullstm_cell_30/Sigmoid_2:y:0!lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_cell_30/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_30_matmul_readvariableop_resource-lstm_cell_30_matmul_1_readvariableop_resource,lstm_cell_30_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1677624*
condR
while_cond_1677623*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
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
:ÿÿÿÿÿÿÿÿÿ´2

IdentityÈ
NoOpNoOp$^lstm_cell_30/BiasAdd/ReadVariableOp#^lstm_cell_30/MatMul/ReadVariableOp%^lstm_cell_30/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ´: : : 2J
#lstm_cell_30/BiasAdd/ReadVariableOp#lstm_cell_30/BiasAdd/ReadVariableOp2H
"lstm_cell_30/MatMul/ReadVariableOp"lstm_cell_30/MatMul/ReadVariableOp2L
$lstm_cell_30/MatMul_1/ReadVariableOp$lstm_cell_30/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs
È
ù
.__inference_lstm_cell_30_layer_call_fn_1681367

inputs
states_0
states_1
unknown:
´Ð
	unknown_0:
´Ð
	unknown_1:	Ð
identity

identity_1

identity_2¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_16757272
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
"
_user_specified_name
states/1
èJ
Õ

lstm_31_while_body_1678555,
(lstm_31_while_lstm_31_while_loop_counter2
.lstm_31_while_lstm_31_while_maximum_iterations
lstm_31_while_placeholder
lstm_31_while_placeholder_1
lstm_31_while_placeholder_2
lstm_31_while_placeholder_3+
'lstm_31_while_lstm_31_strided_slice_1_0g
clstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0:
´ÐQ
=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0:
´ÐK
<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0:	Ð
lstm_31_while_identity
lstm_31_while_identity_1
lstm_31_while_identity_2
lstm_31_while_identity_3
lstm_31_while_identity_4
lstm_31_while_identity_5)
%lstm_31_while_lstm_31_strided_slice_1e
alstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensorM
9lstm_31_while_lstm_cell_31_matmul_readvariableop_resource:
´ÐO
;lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource:
´ÐI
:lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource:	Ð¢1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp¢0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp¢2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpÓ
?lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   2A
?lstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_31/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0lstm_31_while_placeholderHlstm_31/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype023
1lstm_31/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOpReadVariableOp;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype022
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp÷
!lstm_31/while/lstm_cell_31/MatMulMatMul8lstm_31/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2#
!lstm_31/while/lstm_cell_31/MatMulè
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpReadVariableOp=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype024
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOpà
#lstm_31/while/lstm_cell_31/MatMul_1MatMullstm_31_while_placeholder_2:lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2%
#lstm_31/while/lstm_cell_31/MatMul_1Ø
lstm_31/while/lstm_cell_31/addAddV2+lstm_31/while/lstm_cell_31/MatMul:product:0-lstm_31/while/lstm_cell_31/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2 
lstm_31/while/lstm_cell_31/addà
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOpReadVariableOp<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype023
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOpå
"lstm_31/while/lstm_cell_31/BiasAddBiasAdd"lstm_31/while/lstm_cell_31/add:z:09lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2$
"lstm_31/while/lstm_cell_31/BiasAdd
*lstm_31/while/lstm_cell_31/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_31/while/lstm_cell_31/split/split_dim¯
 lstm_31/while/lstm_cell_31/splitSplit3lstm_31/while/lstm_cell_31/split/split_dim:output:0+lstm_31/while/lstm_cell_31/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2"
 lstm_31/while/lstm_cell_31/split±
"lstm_31/while/lstm_cell_31/SigmoidSigmoid)lstm_31/while/lstm_cell_31/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2$
"lstm_31/while/lstm_cell_31/Sigmoidµ
$lstm_31/while/lstm_cell_31/Sigmoid_1Sigmoid)lstm_31/while/lstm_cell_31/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_31/while/lstm_cell_31/Sigmoid_1Á
lstm_31/while/lstm_cell_31/mulMul(lstm_31/while/lstm_cell_31/Sigmoid_1:y:0lstm_31_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2 
lstm_31/while/lstm_cell_31/mul¨
lstm_31/while/lstm_cell_31/ReluRelu)lstm_31/while/lstm_cell_31/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2!
lstm_31/while/lstm_cell_31/ReluÕ
 lstm_31/while/lstm_cell_31/mul_1Mul&lstm_31/while/lstm_cell_31/Sigmoid:y:0-lstm_31/while/lstm_cell_31/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_31/while/lstm_cell_31/mul_1Ê
 lstm_31/while/lstm_cell_31/add_1AddV2"lstm_31/while/lstm_cell_31/mul:z:0$lstm_31/while/lstm_cell_31/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_31/while/lstm_cell_31/add_1µ
$lstm_31/while/lstm_cell_31/Sigmoid_2Sigmoid)lstm_31/while/lstm_cell_31/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2&
$lstm_31/while/lstm_cell_31/Sigmoid_2§
!lstm_31/while/lstm_cell_31/Relu_1Relu$lstm_31/while/lstm_cell_31/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2#
!lstm_31/while/lstm_cell_31/Relu_1Ù
 lstm_31/while/lstm_cell_31/mul_2Mul(lstm_31/while/lstm_cell_31/Sigmoid_2:y:0/lstm_31/while/lstm_cell_31/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2"
 lstm_31/while/lstm_cell_31/mul_2
2lstm_31/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_31_while_placeholder_1lstm_31_while_placeholder$lstm_31/while/lstm_cell_31/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_31/while/TensorArrayV2Write/TensorListSetIteml
lstm_31/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_31/while/add/y
lstm_31/while/addAddV2lstm_31_while_placeholderlstm_31/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_31/while/addp
lstm_31/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_31/while/add_1/y
lstm_31/while/add_1AddV2(lstm_31_while_lstm_31_while_loop_counterlstm_31/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_31/while/add_1
lstm_31/while/IdentityIdentitylstm_31/while/add_1:z:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity¦
lstm_31/while/Identity_1Identity.lstm_31_while_lstm_31_while_maximum_iterations^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_1
lstm_31/while/Identity_2Identitylstm_31/while/add:z:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_2º
lstm_31/while/Identity_3IdentityBlstm_31/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_31/while/NoOp*
T0*
_output_shapes
: 2
lstm_31/while/Identity_3®
lstm_31/while/Identity_4Identity$lstm_31/while/lstm_cell_31/mul_2:z:0^lstm_31/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/while/Identity_4®
lstm_31/while/Identity_5Identity$lstm_31/while/lstm_cell_31/add_1:z:0^lstm_31/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
lstm_31/while/Identity_5
lstm_31/while/NoOpNoOp2^lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp1^lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp3^lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_31/while/NoOp"9
lstm_31_while_identitylstm_31/while/Identity:output:0"=
lstm_31_while_identity_1!lstm_31/while/Identity_1:output:0"=
lstm_31_while_identity_2!lstm_31/while/Identity_2:output:0"=
lstm_31_while_identity_3!lstm_31/while/Identity_3:output:0"=
lstm_31_while_identity_4!lstm_31/while/Identity_4:output:0"=
lstm_31_while_identity_5!lstm_31/while/Identity_5:output:0"P
%lstm_31_while_lstm_31_strided_slice_1'lstm_31_while_lstm_31_strided_slice_1_0"z
:lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource<lstm_31_while_lstm_cell_31_biasadd_readvariableop_resource_0"|
;lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource=lstm_31_while_lstm_cell_31_matmul_1_readvariableop_resource_0"x
9lstm_31_while_lstm_cell_31_matmul_readvariableop_resource;lstm_31_while_lstm_cell_31_matmul_readvariableop_resource_0"È
alstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensorclstm_31_while_tensorarrayv2read_tensorlistgetitem_lstm_31_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2f
1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp1lstm_31/while/lstm_cell_31/BiasAdd/ReadVariableOp2d
0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp0lstm_31/while/lstm_cell_31/MatMul/ReadVariableOp2h
2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp2lstm_31/while/lstm_cell_31/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: 
³?
Õ
while_body_1679946
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_30_matmul_readvariableop_resource_0:
´ÐI
5while_lstm_cell_30_matmul_1_readvariableop_resource_0:
´ÐC
4while_lstm_cell_30_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_30_matmul_readvariableop_resource:
´ÐG
3while_lstm_cell_30_matmul_1_readvariableop_resource:
´ÐA
2while_lstm_cell_30_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_30/BiasAdd/ReadVariableOp¢(while/lstm_cell_30/MatMul/ReadVariableOp¢*while/lstm_cell_30/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ´   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_30/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_30_matmul_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02*
(while/lstm_cell_30/MatMul/ReadVariableOp×
while/lstm_cell_30/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMulÐ
*while/lstm_cell_30/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_30_matmul_1_readvariableop_resource_0* 
_output_shapes
:
´Ð*
dtype02,
*while/lstm_cell_30/MatMul_1/ReadVariableOpÀ
while/lstm_cell_30/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_30/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/MatMul_1¸
while/lstm_cell_30/addAddV2#while/lstm_cell_30/MatMul:product:0%while/lstm_cell_30/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/addÈ
)while/lstm_cell_30/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_30_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_30/BiasAdd/ReadVariableOpÅ
while/lstm_cell_30/BiasAddBiasAddwhile/lstm_cell_30/add:z:01while/lstm_cell_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_30/BiasAdd
"while/lstm_cell_30/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_30/split/split_dim
while/lstm_cell_30/splitSplit+while/lstm_cell_30/split/split_dim:output:0#while/lstm_cell_30/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´*
	num_split2
while/lstm_cell_30/split
while/lstm_cell_30/SigmoidSigmoid!while/lstm_cell_30/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid
while/lstm_cell_30/Sigmoid_1Sigmoid!while/lstm_cell_30/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_1¡
while/lstm_cell_30/mulMul while/lstm_cell_30/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul
while/lstm_cell_30/ReluRelu!while/lstm_cell_30/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Reluµ
while/lstm_cell_30/mul_1Mulwhile/lstm_cell_30/Sigmoid:y:0%while/lstm_cell_30/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_1ª
while/lstm_cell_30/add_1AddV2while/lstm_cell_30/mul:z:0while/lstm_cell_30/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/add_1
while/lstm_cell_30/Sigmoid_2Sigmoid!while/lstm_cell_30/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Sigmoid_2
while/lstm_cell_30/Relu_1Reluwhile/lstm_cell_30/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/Relu_1¹
while/lstm_cell_30/mul_2Mul while/lstm_cell_30/Sigmoid_2:y:0'while/lstm_cell_30/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/lstm_cell_30/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_30/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_30/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_30/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_30/BiasAdd/ReadVariableOp)^while/lstm_cell_30/MatMul/ReadVariableOp+^while/lstm_cell_30/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_30_biasadd_readvariableop_resource4while_lstm_cell_30_biasadd_readvariableop_resource_0"l
3while_lstm_cell_30_matmul_1_readvariableop_resource5while_lstm_cell_30_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_30_matmul_readvariableop_resource3while_lstm_cell_30_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ´:ÿÿÿÿÿÿÿÿÿ´: : : : : 2V
)while/lstm_cell_30/BiasAdd/ReadVariableOp)while/lstm_cell_30/BiasAdd/ReadVariableOp2T
(while/lstm_cell_30/MatMul/ReadVariableOp(while/lstm_cell_30/MatMul/ReadVariableOp2X
*while/lstm_cell_30/MatMul_1/ReadVariableOp*while/lstm_cell_30/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ´:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´:

_output_shapes
: :

_output_shapes
: "¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
K
lstm_29_input:
serving_default_lstm_29_input:0ÿÿÿÿÿÿÿÿÿ@
dense_324
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¼
à
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
Ä__call__
+Å&call_and_return_all_conditional_losses
Æ_default_save_signature"
_tf_keras_sequential
Å
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
trainable_variables
	variables
regularization_losses
	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"
_tf_keras_layer
Å
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
 trainable_variables
!	variables
"regularization_losses
#	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"
_tf_keras_layer
Å
$cell
%
state_spec
&regularization_losses
'	variables
(trainable_variables
)	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
*trainable_variables
+	variables
,regularization_losses
-	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"
_tf_keras_layer
½

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"
_tf_keras_layer
§
4trainable_variables
5	variables
6regularization_losses
7	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
½

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"
_tf_keras_layer
×
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate.mª/m«8m¬9m­Cm®Dm¯Em°Fm±Gm²Hm³Im´JmµKm¶.v·/v¸8v¹9vºCv»Dv¼Ev½Fv¾Gv¿HvÀIvÁJvÂKvÃ"
	optimizer
 "
trackable_list_wrapper
~
C0
D1
E2
F3
G4
H5
I6
J7
K8
.9
/10
811
912"
trackable_list_wrapper
~
C0
D1
E2
F3
G4
H5
I6
J7
K8
.9
/10
811
912"
trackable_list_wrapper
Î
regularization_losses
Llayer_regularization_losses
Mnon_trainable_variables
	variables
Nlayer_metrics
trainable_variables
Ometrics

Players
Ä__call__
Æ_default_save_signature
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
-
Ùserving_default"
signature_map
ã
Q
state_size

Ckernel
Drecurrent_kernel
Ebias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
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
¼
regularization_losses
Vlayer_regularization_losses

Wstates
Xnon_trainable_variables
	variables
Ylayer_metrics
trainable_variables
Zmetrics

[layers
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
\layer_regularization_losses
]non_trainable_variables
trainable_variables
	variables
^layer_metrics
regularization_losses
_metrics

`layers
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
ã
a
state_size

Fkernel
Grecurrent_kernel
Hbias
btrainable_variables
c	variables
dregularization_losses
e	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"
_tf_keras_layer
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
¼
regularization_losses
flayer_regularization_losses

gstates
hnon_trainable_variables
	variables
ilayer_metrics
trainable_variables
jmetrics

klayers
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
llayer_regularization_losses
mnon_trainable_variables
 trainable_variables
!	variables
nlayer_metrics
"regularization_losses
ometrics

players
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
ã
q
state_size

Ikernel
Jrecurrent_kernel
Kbias
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
¼
&regularization_losses
vlayer_regularization_losses

wstates
xnon_trainable_variables
'	variables
ylayer_metrics
(trainable_variables
zmetrics

{layers
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
±
|layer_regularization_losses
}non_trainable_variables
*trainable_variables
+	variables
~layer_metrics
,regularization_losses
metrics
layers
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
#:!
´´2dense_31/kernel
:´2dense_31/bias
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
µ
 layer_regularization_losses
non_trainable_variables
0trainable_variables
1	variables
layer_metrics
2regularization_losses
metrics
layers
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
non_trainable_variables
4trainable_variables
5	variables
layer_metrics
6regularization_losses
metrics
layers
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
": 	´2dense_32/kernel
:2dense_32/bias
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
µ
 layer_regularization_losses
non_trainable_variables
:trainable_variables
;	variables
layer_metrics
<regularization_losses
metrics
layers
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	Ð2lstm_29/lstm_cell_29/kernel
9:7
´Ð2%lstm_29/lstm_cell_29/recurrent_kernel
(:&Ð2lstm_29/lstm_cell_29/bias
/:-
´Ð2lstm_30/lstm_cell_30/kernel
9:7
´Ð2%lstm_30/lstm_cell_30/recurrent_kernel
(:&Ð2lstm_30/lstm_cell_30/bias
/:-
´Ð2lstm_31/lstm_cell_31/kernel
9:7
´Ð2%lstm_31/lstm_cell_31/recurrent_kernel
(:&Ð2lstm_31/lstm_cell_31/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
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
µ
 layer_regularization_losses
non_trainable_variables
Rtrainable_variables
S	variables
layer_metrics
Tregularization_losses
metrics
layers
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
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
'
0"
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
µ
 layer_regularization_losses
non_trainable_variables
btrainable_variables
c	variables
layer_metrics
dregularization_losses
metrics
layers
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
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
'
0"
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
5
I0
J1
K2"
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
non_trainable_variables
rtrainable_variables
s	variables
layer_metrics
tregularization_losses
metrics
 layers
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
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
'
$0"
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

¡total

¢count
£	variables
¤	keras_api"
_tf_keras_metric
c

¥total

¦count
§
_fn_kwargs
¨	variables
©	keras_api"
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
¡0
¢1"
trackable_list_wrapper
.
£	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¥0
¦1"
trackable_list_wrapper
.
¨	variables"
_generic_user_object
(:&
´´2Adam/dense_31/kernel/m
!:´2Adam/dense_31/bias/m
':%	´2Adam/dense_32/kernel/m
 :2Adam/dense_32/bias/m
3:1	Ð2"Adam/lstm_29/lstm_cell_29/kernel/m
>:<
´Ð2,Adam/lstm_29/lstm_cell_29/recurrent_kernel/m
-:+Ð2 Adam/lstm_29/lstm_cell_29/bias/m
4:2
´Ð2"Adam/lstm_30/lstm_cell_30/kernel/m
>:<
´Ð2,Adam/lstm_30/lstm_cell_30/recurrent_kernel/m
-:+Ð2 Adam/lstm_30/lstm_cell_30/bias/m
4:2
´Ð2"Adam/lstm_31/lstm_cell_31/kernel/m
>:<
´Ð2,Adam/lstm_31/lstm_cell_31/recurrent_kernel/m
-:+Ð2 Adam/lstm_31/lstm_cell_31/bias/m
(:&
´´2Adam/dense_31/kernel/v
!:´2Adam/dense_31/bias/v
':%	´2Adam/dense_32/kernel/v
 :2Adam/dense_32/bias/v
3:1	Ð2"Adam/lstm_29/lstm_cell_29/kernel/v
>:<
´Ð2,Adam/lstm_29/lstm_cell_29/recurrent_kernel/v
-:+Ð2 Adam/lstm_29/lstm_cell_29/bias/v
4:2
´Ð2"Adam/lstm_30/lstm_cell_30/kernel/v
>:<
´Ð2,Adam/lstm_30/lstm_cell_30/recurrent_kernel/v
-:+Ð2 Adam/lstm_30/lstm_cell_30/bias/v
4:2
´Ð2"Adam/lstm_31/lstm_cell_31/kernel/v
>:<
´Ð2,Adam/lstm_31/lstm_cell_31/recurrent_kernel/v
-:+Ð2 Adam/lstm_31/lstm_cell_31/bias/v
2
/__inference_sequential_13_layer_call_fn_1677298
/__inference_sequential_13_layer_call_fn_1678185
/__inference_sequential_13_layer_call_fn_1678216
/__inference_sequential_13_layer_call_fn_1678037À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
J__inference_sequential_13_layer_call_and_return_conditional_losses_1678694
J__inference_sequential_13_layer_call_and_return_conditional_losses_1679200
J__inference_sequential_13_layer_call_and_return_conditional_losses_1678076
J__inference_sequential_13_layer_call_and_return_conditional_losses_1678115À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÓBÐ
"__inference__wrapped_model_1674916lstm_29_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
)__inference_lstm_29_layer_call_fn_1679211
)__inference_lstm_29_layer_call_fn_1679222
)__inference_lstm_29_layer_call_fn_1679233
)__inference_lstm_29_layer_call_fn_1679244Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
D__inference_lstm_29_layer_call_and_return_conditional_losses_1679387
D__inference_lstm_29_layer_call_and_return_conditional_losses_1679530
D__inference_lstm_29_layer_call_and_return_conditional_losses_1679673
D__inference_lstm_29_layer_call_and_return_conditional_losses_1679816Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
,__inference_dropout_47_layer_call_fn_1679821
,__inference_dropout_47_layer_call_fn_1679826´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
G__inference_dropout_47_layer_call_and_return_conditional_losses_1679831
G__inference_dropout_47_layer_call_and_return_conditional_losses_1679843´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_lstm_30_layer_call_fn_1679854
)__inference_lstm_30_layer_call_fn_1679865
)__inference_lstm_30_layer_call_fn_1679876
)__inference_lstm_30_layer_call_fn_1679887Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
D__inference_lstm_30_layer_call_and_return_conditional_losses_1680030
D__inference_lstm_30_layer_call_and_return_conditional_losses_1680173
D__inference_lstm_30_layer_call_and_return_conditional_losses_1680316
D__inference_lstm_30_layer_call_and_return_conditional_losses_1680459Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
,__inference_dropout_48_layer_call_fn_1680464
,__inference_dropout_48_layer_call_fn_1680469´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
G__inference_dropout_48_layer_call_and_return_conditional_losses_1680474
G__inference_dropout_48_layer_call_and_return_conditional_losses_1680486´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_lstm_31_layer_call_fn_1680497
)__inference_lstm_31_layer_call_fn_1680508
)__inference_lstm_31_layer_call_fn_1680519
)__inference_lstm_31_layer_call_fn_1680530Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
D__inference_lstm_31_layer_call_and_return_conditional_losses_1680673
D__inference_lstm_31_layer_call_and_return_conditional_losses_1680816
D__inference_lstm_31_layer_call_and_return_conditional_losses_1680959
D__inference_lstm_31_layer_call_and_return_conditional_losses_1681102Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
,__inference_dropout_49_layer_call_fn_1681107
,__inference_dropout_49_layer_call_fn_1681112´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
G__inference_dropout_49_layer_call_and_return_conditional_losses_1681117
G__inference_dropout_49_layer_call_and_return_conditional_losses_1681129´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
*__inference_dense_31_layer_call_fn_1681138¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_31_layer_call_and_return_conditional_losses_1681169¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
,__inference_dropout_50_layer_call_fn_1681174
,__inference_dropout_50_layer_call_fn_1681179´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
G__inference_dropout_50_layer_call_and_return_conditional_losses_1681184
G__inference_dropout_50_layer_call_and_return_conditional_losses_1681196´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
*__inference_dense_32_layer_call_fn_1681205¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_dense_32_layer_call_and_return_conditional_losses_1681235¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÒBÏ
%__inference_signature_wrapper_1678154lstm_29_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¤2¡
.__inference_lstm_cell_29_layer_call_fn_1681252
.__inference_lstm_cell_29_layer_call_fn_1681269¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1681301
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1681333¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¤2¡
.__inference_lstm_cell_30_layer_call_fn_1681350
.__inference_lstm_cell_30_layer_call_fn_1681367¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1681399
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1681431¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¤2¡
.__inference_lstm_cell_31_layer_call_fn_1681448
.__inference_lstm_cell_31_layer_call_fn_1681465¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1681497
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1681529¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 «
"__inference__wrapped_model_1674916CDEFGHIJK./89:¢7
0¢-
+(
lstm_29_inputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2
dense_32&#
dense_32ÿÿÿÿÿÿÿÿÿ¯
E__inference_dense_31_layer_call_and_return_conditional_losses_1681169f./4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ´
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 
*__inference_dense_31_layer_call_fn_1681138Y./4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ´
ª "ÿÿÿÿÿÿÿÿÿ´®
E__inference_dense_32_layer_call_and_return_conditional_losses_1681235e894¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ´
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_32_layer_call_fn_1681205X894¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ´
ª "ÿÿÿÿÿÿÿÿÿ±
G__inference_dropout_47_layer_call_and_return_conditional_losses_1679831f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 ±
G__inference_dropout_47_layer_call_and_return_conditional_losses_1679843f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 
,__inference_dropout_47_layer_call_fn_1679821Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p 
ª "ÿÿÿÿÿÿÿÿÿ´
,__inference_dropout_47_layer_call_fn_1679826Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p
ª "ÿÿÿÿÿÿÿÿÿ´±
G__inference_dropout_48_layer_call_and_return_conditional_losses_1680474f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 ±
G__inference_dropout_48_layer_call_and_return_conditional_losses_1680486f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 
,__inference_dropout_48_layer_call_fn_1680464Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p 
ª "ÿÿÿÿÿÿÿÿÿ´
,__inference_dropout_48_layer_call_fn_1680469Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p
ª "ÿÿÿÿÿÿÿÿÿ´±
G__inference_dropout_49_layer_call_and_return_conditional_losses_1681117f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 ±
G__inference_dropout_49_layer_call_and_return_conditional_losses_1681129f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 
,__inference_dropout_49_layer_call_fn_1681107Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p 
ª "ÿÿÿÿÿÿÿÿÿ´
,__inference_dropout_49_layer_call_fn_1681112Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p
ª "ÿÿÿÿÿÿÿÿÿ´±
G__inference_dropout_50_layer_call_and_return_conditional_losses_1681184f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 ±
G__inference_dropout_50_layer_call_and_return_conditional_losses_1681196f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 
,__inference_dropout_50_layer_call_fn_1681174Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p 
ª "ÿÿÿÿÿÿÿÿÿ´
,__inference_dropout_50_layer_call_fn_1681179Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ´
p
ª "ÿÿÿÿÿÿÿÿÿ´Ô
D__inference_lstm_29_layer_call_and_return_conditional_losses_1679387CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
 Ô
D__inference_lstm_29_layer_call_and_return_conditional_losses_1679530CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
 º
D__inference_lstm_29_layer_call_and_return_conditional_losses_1679673rCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 º
D__inference_lstm_29_layer_call_and_return_conditional_losses_1679816rCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 «
)__inference_lstm_29_layer_call_fn_1679211~CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´«
)__inference_lstm_29_layer_call_fn_1679222~CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
)__inference_lstm_29_layer_call_fn_1679233eCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ´
)__inference_lstm_29_layer_call_fn_1679244eCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ´Õ
D__inference_lstm_30_layer_call_and_return_conditional_losses_1680030FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
 Õ
D__inference_lstm_30_layer_call_and_return_conditional_losses_1680173FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
 »
D__inference_lstm_30_layer_call_and_return_conditional_losses_1680316sFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ´

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 »
D__inference_lstm_30_layer_call_and_return_conditional_losses_1680459sFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ´

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 ¬
)__inference_lstm_30_layer_call_fn_1679854FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´¬
)__inference_lstm_30_layer_call_fn_1679865FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
)__inference_lstm_30_layer_call_fn_1679876fFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ´

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ´
)__inference_lstm_30_layer_call_fn_1679887fFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ´

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ´Õ
D__inference_lstm_31_layer_call_and_return_conditional_losses_1680673IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
 Õ
D__inference_lstm_31_layer_call_and_return_conditional_losses_1680816IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
 »
D__inference_lstm_31_layer_call_and_return_conditional_losses_1680959sIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ´

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 »
D__inference_lstm_31_layer_call_and_return_conditional_losses_1681102sIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ´

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ´
 ¬
)__inference_lstm_31_layer_call_fn_1680497IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´¬
)__inference_lstm_31_layer_call_fn_1680508IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
)__inference_lstm_31_layer_call_fn_1680519fIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ´

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ´
)__inference_lstm_31_layer_call_fn_1680530fIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ´

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ´Ð
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1681301CDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ´
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ´
 
0/1/1ÿÿÿÿÿÿÿÿÿ´
 Ð
I__inference_lstm_cell_29_layer_call_and_return_conditional_losses_1681333CDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ´
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ´
 
0/1/1ÿÿÿÿÿÿÿÿÿ´
 ¥
.__inference_lstm_cell_29_layer_call_fn_1681252òCDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ´
C@

1/0ÿÿÿÿÿÿÿÿÿ´

1/1ÿÿÿÿÿÿÿÿÿ´¥
.__inference_lstm_cell_29_layer_call_fn_1681269òCDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ´
C@

1/0ÿÿÿÿÿÿÿÿÿ´

1/1ÿÿÿÿÿÿÿÿÿ´Ò
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1681399FGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ´
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ´
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ´
 
0/1/1ÿÿÿÿÿÿÿÿÿ´
 Ò
I__inference_lstm_cell_30_layer_call_and_return_conditional_losses_1681431FGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ´
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ´
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ´
 
0/1/1ÿÿÿÿÿÿÿÿÿ´
 §
.__inference_lstm_cell_30_layer_call_fn_1681350ôFGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ´
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ´
C@

1/0ÿÿÿÿÿÿÿÿÿ´

1/1ÿÿÿÿÿÿÿÿÿ´§
.__inference_lstm_cell_30_layer_call_fn_1681367ôFGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ´
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ´
C@

1/0ÿÿÿÿÿÿÿÿÿ´

1/1ÿÿÿÿÿÿÿÿÿ´Ò
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1681497IJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ´
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ´
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ´
 
0/1/1ÿÿÿÿÿÿÿÿÿ´
 Ò
I__inference_lstm_cell_31_layer_call_and_return_conditional_losses_1681529IJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ´
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ´
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ´
 
0/1/1ÿÿÿÿÿÿÿÿÿ´
 §
.__inference_lstm_cell_31_layer_call_fn_1681448ôIJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ´
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ´
C@

1/0ÿÿÿÿÿÿÿÿÿ´

1/1ÿÿÿÿÿÿÿÿÿ´§
.__inference_lstm_cell_31_layer_call_fn_1681465ôIJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ´
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ´
# 
states/1ÿÿÿÿÿÿÿÿÿ´
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ´
C@

1/0ÿÿÿÿÿÿÿÿÿ´

1/1ÿÿÿÿÿÿÿÿÿ´Ì
J__inference_sequential_13_layer_call_and_return_conditional_losses_1678076~CDEFGHIJK./89B¢?
8¢5
+(
lstm_29_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ì
J__inference_sequential_13_layer_call_and_return_conditional_losses_1678115~CDEFGHIJK./89B¢?
8¢5
+(
lstm_29_inputÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Å
J__inference_sequential_13_layer_call_and_return_conditional_losses_1678694wCDEFGHIJK./89;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Å
J__inference_sequential_13_layer_call_and_return_conditional_losses_1679200wCDEFGHIJK./89;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 ¤
/__inference_sequential_13_layer_call_fn_1677298qCDEFGHIJK./89B¢?
8¢5
+(
lstm_29_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
/__inference_sequential_13_layer_call_fn_1678037qCDEFGHIJK./89B¢?
8¢5
+(
lstm_29_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_13_layer_call_fn_1678185jCDEFGHIJK./89;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_13_layer_call_fn_1678216jCDEFGHIJK./89;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¿
%__inference_signature_wrapper_1678154CDEFGHIJK./89K¢H
¢ 
Aª>
<
lstm_29_input+(
lstm_29_inputÿÿÿÿÿÿÿÿÿ"7ª4
2
dense_32&#
dense_32ÿÿÿÿÿÿÿÿÿ