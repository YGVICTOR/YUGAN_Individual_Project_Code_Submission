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
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ªª* 
shared_namedense_29/kernel
u
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel* 
_output_shapes
:
ªª*
dtype0
s
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ª*
shared_namedense_29/bias
l
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes	
:ª*
dtype0
{
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ª* 
shared_namedense_30/kernel
t
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes
:	ª*
dtype0
r
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_30/bias
k
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
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
lstm_26/lstm_cell_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¨*,
shared_namelstm_26/lstm_cell_26/kernel

/lstm_26/lstm_cell_26/kernel/Read/ReadVariableOpReadVariableOplstm_26/lstm_cell_26/kernel*
_output_shapes
:	¨*
dtype0
¨
%lstm_26/lstm_cell_26/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*6
shared_name'%lstm_26/lstm_cell_26/recurrent_kernel
¡
9lstm_26/lstm_cell_26/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_26/lstm_cell_26/recurrent_kernel* 
_output_shapes
:
ª¨*
dtype0

lstm_26/lstm_cell_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨**
shared_namelstm_26/lstm_cell_26/bias

-lstm_26/lstm_cell_26/bias/Read/ReadVariableOpReadVariableOplstm_26/lstm_cell_26/bias*
_output_shapes	
:¨*
dtype0

lstm_27/lstm_cell_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*,
shared_namelstm_27/lstm_cell_27/kernel

/lstm_27/lstm_cell_27/kernel/Read/ReadVariableOpReadVariableOplstm_27/lstm_cell_27/kernel* 
_output_shapes
:
ª¨*
dtype0
¨
%lstm_27/lstm_cell_27/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*6
shared_name'%lstm_27/lstm_cell_27/recurrent_kernel
¡
9lstm_27/lstm_cell_27/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_27/lstm_cell_27/recurrent_kernel* 
_output_shapes
:
ª¨*
dtype0

lstm_27/lstm_cell_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨**
shared_namelstm_27/lstm_cell_27/bias

-lstm_27/lstm_cell_27/bias/Read/ReadVariableOpReadVariableOplstm_27/lstm_cell_27/bias*
_output_shapes	
:¨*
dtype0

lstm_28/lstm_cell_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*,
shared_namelstm_28/lstm_cell_28/kernel

/lstm_28/lstm_cell_28/kernel/Read/ReadVariableOpReadVariableOplstm_28/lstm_cell_28/kernel* 
_output_shapes
:
ª¨*
dtype0
¨
%lstm_28/lstm_cell_28/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*6
shared_name'%lstm_28/lstm_cell_28/recurrent_kernel
¡
9lstm_28/lstm_cell_28/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_28/lstm_cell_28/recurrent_kernel* 
_output_shapes
:
ª¨*
dtype0

lstm_28/lstm_cell_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨**
shared_namelstm_28/lstm_cell_28/bias

-lstm_28/lstm_cell_28/bias/Read/ReadVariableOpReadVariableOplstm_28/lstm_cell_28/bias*
_output_shapes	
:¨*
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
Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ªª*'
shared_nameAdam/dense_29/kernel/m

*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m* 
_output_shapes
:
ªª*
dtype0

Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ª*%
shared_nameAdam/dense_29/bias/m
z
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
_output_shapes	
:ª*
dtype0

Adam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ª*'
shared_nameAdam/dense_30/kernel/m

*Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/m*
_output_shapes
:	ª*
dtype0

Adam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_30/bias/m
y
(Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/m*
_output_shapes
:*
dtype0
¡
"Adam/lstm_26/lstm_cell_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¨*3
shared_name$"Adam/lstm_26/lstm_cell_26/kernel/m

6Adam/lstm_26/lstm_cell_26/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_26/lstm_cell_26/kernel/m*
_output_shapes
:	¨*
dtype0
¶
,Adam/lstm_26/lstm_cell_26/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*=
shared_name.,Adam/lstm_26/lstm_cell_26/recurrent_kernel/m
¯
@Adam/lstm_26/lstm_cell_26/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_26/lstm_cell_26/recurrent_kernel/m* 
_output_shapes
:
ª¨*
dtype0

 Adam/lstm_26/lstm_cell_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨*1
shared_name" Adam/lstm_26/lstm_cell_26/bias/m

4Adam/lstm_26/lstm_cell_26/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_26/lstm_cell_26/bias/m*
_output_shapes	
:¨*
dtype0
¢
"Adam/lstm_27/lstm_cell_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*3
shared_name$"Adam/lstm_27/lstm_cell_27/kernel/m

6Adam/lstm_27/lstm_cell_27/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_27/lstm_cell_27/kernel/m* 
_output_shapes
:
ª¨*
dtype0
¶
,Adam/lstm_27/lstm_cell_27/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*=
shared_name.,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m
¯
@Adam/lstm_27/lstm_cell_27/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m* 
_output_shapes
:
ª¨*
dtype0

 Adam/lstm_27/lstm_cell_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨*1
shared_name" Adam/lstm_27/lstm_cell_27/bias/m

4Adam/lstm_27/lstm_cell_27/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_27/lstm_cell_27/bias/m*
_output_shapes	
:¨*
dtype0
¢
"Adam/lstm_28/lstm_cell_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*3
shared_name$"Adam/lstm_28/lstm_cell_28/kernel/m

6Adam/lstm_28/lstm_cell_28/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_28/lstm_cell_28/kernel/m* 
_output_shapes
:
ª¨*
dtype0
¶
,Adam/lstm_28/lstm_cell_28/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*=
shared_name.,Adam/lstm_28/lstm_cell_28/recurrent_kernel/m
¯
@Adam/lstm_28/lstm_cell_28/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_28/lstm_cell_28/recurrent_kernel/m* 
_output_shapes
:
ª¨*
dtype0

 Adam/lstm_28/lstm_cell_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨*1
shared_name" Adam/lstm_28/lstm_cell_28/bias/m

4Adam/lstm_28/lstm_cell_28/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_28/lstm_cell_28/bias/m*
_output_shapes	
:¨*
dtype0

Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ªª*'
shared_nameAdam/dense_29/kernel/v

*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v* 
_output_shapes
:
ªª*
dtype0

Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ª*%
shared_nameAdam/dense_29/bias/v
z
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
_output_shapes	
:ª*
dtype0

Adam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ª*'
shared_nameAdam/dense_30/kernel/v

*Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/v*
_output_shapes
:	ª*
dtype0

Adam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_30/bias/v
y
(Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/v*
_output_shapes
:*
dtype0
¡
"Adam/lstm_26/lstm_cell_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¨*3
shared_name$"Adam/lstm_26/lstm_cell_26/kernel/v

6Adam/lstm_26/lstm_cell_26/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_26/lstm_cell_26/kernel/v*
_output_shapes
:	¨*
dtype0
¶
,Adam/lstm_26/lstm_cell_26/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*=
shared_name.,Adam/lstm_26/lstm_cell_26/recurrent_kernel/v
¯
@Adam/lstm_26/lstm_cell_26/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_26/lstm_cell_26/recurrent_kernel/v* 
_output_shapes
:
ª¨*
dtype0

 Adam/lstm_26/lstm_cell_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨*1
shared_name" Adam/lstm_26/lstm_cell_26/bias/v

4Adam/lstm_26/lstm_cell_26/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_26/lstm_cell_26/bias/v*
_output_shapes	
:¨*
dtype0
¢
"Adam/lstm_27/lstm_cell_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*3
shared_name$"Adam/lstm_27/lstm_cell_27/kernel/v

6Adam/lstm_27/lstm_cell_27/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_27/lstm_cell_27/kernel/v* 
_output_shapes
:
ª¨*
dtype0
¶
,Adam/lstm_27/lstm_cell_27/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*=
shared_name.,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v
¯
@Adam/lstm_27/lstm_cell_27/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v* 
_output_shapes
:
ª¨*
dtype0

 Adam/lstm_27/lstm_cell_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨*1
shared_name" Adam/lstm_27/lstm_cell_27/bias/v

4Adam/lstm_27/lstm_cell_27/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_27/lstm_cell_27/bias/v*
_output_shapes	
:¨*
dtype0
¢
"Adam/lstm_28/lstm_cell_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*3
shared_name$"Adam/lstm_28/lstm_cell_28/kernel/v

6Adam/lstm_28/lstm_cell_28/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_28/lstm_cell_28/kernel/v* 
_output_shapes
:
ª¨*
dtype0
¶
,Adam/lstm_28/lstm_cell_28/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ª¨*=
shared_name.,Adam/lstm_28/lstm_cell_28/recurrent_kernel/v
¯
@Adam/lstm_28/lstm_cell_28/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_28/lstm_cell_28/recurrent_kernel/v* 
_output_shapes
:
ª¨*
dtype0

 Adam/lstm_28/lstm_cell_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨*1
shared_name" Adam/lstm_28/lstm_cell_28/bias/v

4Adam/lstm_28/lstm_cell_28/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_28/lstm_cell_28/bias/v*
_output_shapes	
:¨*
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
l
cell

state_spec
	variables
regularization_losses
trainable_variables
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
regularization_losses
trainable_variables
	keras_api
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
l
$cell
%
state_spec
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
Ä
>iter

?beta_1

@beta_2
	Adecay
Blearning_rate.mª/m«8m¬9m­Cm®Dm¯Em°Fm±Gm²Hm³Im´JmµKm¶.v·/v¸8v¹9vºCv»Dv¼Ev½Fv¾Gv¿HvÀIvÁJvÂKvÃ
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
­
	variables
regularization_losses
Lmetrics
Mlayer_metrics

Nlayers
Olayer_regularization_losses
trainable_variables
Pnon_trainable_variables
 

Q
state_size

Ckernel
Drecurrent_kernel
Ebias
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
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
¹
	variables
regularization_losses
Vmetrics
Wlayer_metrics

Xlayers
Ylayer_regularization_losses
trainable_variables
Znon_trainable_variables

[states
 
 
 
­
	variables
regularization_losses
\metrics
]layer_metrics

^layers
_layer_regularization_losses
trainable_variables
`non_trainable_variables

a
state_size

Fkernel
Grecurrent_kernel
Hbias
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
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
¹
	variables
regularization_losses
fmetrics
glayer_metrics

hlayers
ilayer_regularization_losses
trainable_variables
jnon_trainable_variables

kstates
 
 
 
­
 	variables
!regularization_losses
lmetrics
mlayer_metrics

nlayers
olayer_regularization_losses
"trainable_variables
pnon_trainable_variables

q
state_size

Ikernel
Jrecurrent_kernel
Kbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
 

I0
J1
K2
 

I0
J1
K2
¹
&	variables
'regularization_losses
vmetrics
wlayer_metrics

xlayers
ylayer_regularization_losses
(trainable_variables
znon_trainable_variables

{states
 
 
 
®
*	variables
+regularization_losses
|metrics
}layer_metrics

~layers
layer_regularization_losses
,trainable_variables
non_trainable_variables
[Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_29/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
²
0	variables
1regularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
2trainable_variables
non_trainable_variables
 
 
 
²
4	variables
5regularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
6trainable_variables
non_trainable_variables
[Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_30/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
²
:	variables
;regularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
<trainable_variables
non_trainable_variables
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
VARIABLE_VALUElstm_26/lstm_cell_26/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_26/lstm_cell_26/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_26/lstm_cell_26/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_27/lstm_cell_27/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_27/lstm_cell_27/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_27/lstm_cell_27/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_28/lstm_cell_28/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_28/lstm_cell_28/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_28/lstm_cell_28/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE

0
1
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
 
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
²
R	variables
Sregularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
Ttrainable_variables
non_trainable_variables
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
²
b	variables
cregularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
dtrainable_variables
non_trainable_variables
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

I0
J1
K2
 

I0
J1
K2
²
r	variables
sregularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
ttrainable_variables
 non_trainable_variables
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
VARIABLE_VALUEAdam/dense_29/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_26/lstm_cell_26/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_26/lstm_cell_26/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_26/lstm_cell_26/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_27/lstm_cell_27/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_27/lstm_cell_27/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_27/lstm_cell_27/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_28/lstm_cell_28/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_28/lstm_cell_28/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_28/lstm_cell_28/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_26/lstm_cell_26/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_26/lstm_cell_26/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_26/lstm_cell_26/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_27/lstm_cell_27/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_27/lstm_cell_27/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_27/lstm_cell_27/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_28/lstm_cell_28/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_28/lstm_cell_28/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_28/lstm_cell_28/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_26_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
´
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_26_inputlstm_26/lstm_cell_26/kernel%lstm_26/lstm_cell_26/recurrent_kernellstm_26/lstm_cell_26/biaslstm_27/lstm_cell_27/kernel%lstm_27/lstm_cell_27/recurrent_kernellstm_27/lstm_cell_27/biaslstm_28/lstm_cell_28/kernel%lstm_28/lstm_cell_28/recurrent_kernellstm_28/lstm_cell_28/biasdense_29/kerneldense_29/biasdense_30/kerneldense_30/bias*
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
%__inference_signature_wrapper_1060879
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¾
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_26/lstm_cell_26/kernel/Read/ReadVariableOp9lstm_26/lstm_cell_26/recurrent_kernel/Read/ReadVariableOp-lstm_26/lstm_cell_26/bias/Read/ReadVariableOp/lstm_27/lstm_cell_27/kernel/Read/ReadVariableOp9lstm_27/lstm_cell_27/recurrent_kernel/Read/ReadVariableOp-lstm_27/lstm_cell_27/bias/Read/ReadVariableOp/lstm_28/lstm_cell_28/kernel/Read/ReadVariableOp9lstm_28/lstm_cell_28/recurrent_kernel/Read/ReadVariableOp-lstm_28/lstm_cell_28/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp*Adam/dense_30/kernel/m/Read/ReadVariableOp(Adam/dense_30/bias/m/Read/ReadVariableOp6Adam/lstm_26/lstm_cell_26/kernel/m/Read/ReadVariableOp@Adam/lstm_26/lstm_cell_26/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_26/lstm_cell_26/bias/m/Read/ReadVariableOp6Adam/lstm_27/lstm_cell_27/kernel/m/Read/ReadVariableOp@Adam/lstm_27/lstm_cell_27/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_27/lstm_cell_27/bias/m/Read/ReadVariableOp6Adam/lstm_28/lstm_cell_28/kernel/m/Read/ReadVariableOp@Adam/lstm_28/lstm_cell_28/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_28/lstm_cell_28/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOp*Adam/dense_30/kernel/v/Read/ReadVariableOp(Adam/dense_30/bias/v/Read/ReadVariableOp6Adam/lstm_26/lstm_cell_26/kernel/v/Read/ReadVariableOp@Adam/lstm_26/lstm_cell_26/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_26/lstm_cell_26/bias/v/Read/ReadVariableOp6Adam/lstm_27/lstm_cell_27/kernel/v/Read/ReadVariableOp@Adam/lstm_27/lstm_cell_27/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_27/lstm_cell_27/bias/v/Read/ReadVariableOp6Adam/lstm_28/lstm_cell_28/kernel/v/Read/ReadVariableOp@Adam/lstm_28/lstm_cell_28/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_28/lstm_cell_28/bias/v/Read/ReadVariableOpConst*=
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
 __inference__traced_save_1064421
ù
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_29/kerneldense_29/biasdense_30/kerneldense_30/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_26/lstm_cell_26/kernel%lstm_26/lstm_cell_26/recurrent_kernellstm_26/lstm_cell_26/biaslstm_27/lstm_cell_27/kernel%lstm_27/lstm_cell_27/recurrent_kernellstm_27/lstm_cell_27/biaslstm_28/lstm_cell_28/kernel%lstm_28/lstm_cell_28/recurrent_kernellstm_28/lstm_cell_28/biastotalcounttotal_1count_1Adam/dense_29/kernel/mAdam/dense_29/bias/mAdam/dense_30/kernel/mAdam/dense_30/bias/m"Adam/lstm_26/lstm_cell_26/kernel/m,Adam/lstm_26/lstm_cell_26/recurrent_kernel/m Adam/lstm_26/lstm_cell_26/bias/m"Adam/lstm_27/lstm_cell_27/kernel/m,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m Adam/lstm_27/lstm_cell_27/bias/m"Adam/lstm_28/lstm_cell_28/kernel/m,Adam/lstm_28/lstm_cell_28/recurrent_kernel/m Adam/lstm_28/lstm_cell_28/bias/mAdam/dense_29/kernel/vAdam/dense_29/bias/vAdam/dense_30/kernel/vAdam/dense_30/bias/v"Adam/lstm_26/lstm_cell_26/kernel/v,Adam/lstm_26/lstm_cell_26/recurrent_kernel/v Adam/lstm_26/lstm_cell_26/bias/v"Adam/lstm_27/lstm_cell_27/kernel/v,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v Adam/lstm_27/lstm_cell_27/bias/v"Adam/lstm_28/lstm_cell_28/kernel/v,Adam/lstm_28/lstm_cell_28/recurrent_kernel/v Adam/lstm_28/lstm_cell_28/bias/v*<
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
#__inference__traced_restore_1064575­È6
Þ
È
while_cond_1062956
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1062956___redundant_placeholder05
1while_while_cond_1062956___redundant_placeholder15
1while_while_cond_1062956___redundant_placeholder25
1while_while_cond_1062956___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
È
ù
.__inference_lstm_cell_28_layer_call_fn_1064190

inputs
states_0
states_1
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
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
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_10590502
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1
Þ
È
while_cond_1063099
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1063099___redundant_placeholder05
1while_while_cond_1063099___redundant_placeholder15
1while_while_cond_1063099___redundant_placeholder25
1while_while_cond_1063099___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
ÃU

D__inference_lstm_27_layer_call_and_return_conditional_losses_1063184

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ª¨A
-lstm_cell_27_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_27_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
:ÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1063100*
condR
while_cond_1063099*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
V
 
D__inference_lstm_27_layer_call_and_return_conditional_losses_1062755
inputs_0?
+lstm_cell_27_matmul_readvariableop_resource:
ª¨A
-lstm_cell_27_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_27_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileF
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1062671*
condR
while_cond_1062670*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
inputs/0
³?
Õ
while_body_1060161
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_28_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_28_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_28_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_28_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_28_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_28/BiasAdd/ReadVariableOp¢(while/lstm_cell_28/MatMul/ReadVariableOp¢*while/lstm_cell_28/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_28/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_28_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_28/MatMul/ReadVariableOp×
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMulÐ
*while/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_28_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_28/MatMul_1/ReadVariableOpÀ
while/lstm_cell_28/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMul_1¸
while/lstm_cell_28/addAddV2#while/lstm_cell_28/MatMul:product:0%while/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/addÈ
)while/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_28_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_28/BiasAdd/ReadVariableOpÅ
while/lstm_cell_28/BiasAddBiasAddwhile/lstm_cell_28/add:z:01while/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/BiasAdd
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_28/split/split_dim
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0#while/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_28/split
while/lstm_cell_28/SigmoidSigmoid!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid
while/lstm_cell_28/Sigmoid_1Sigmoid!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_1¡
while/lstm_cell_28/mulMul while/lstm_cell_28/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul
while/lstm_cell_28/ReluRelu!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Reluµ
while/lstm_cell_28/mul_1Mulwhile/lstm_cell_28/Sigmoid:y:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_1ª
while/lstm_cell_28/add_1AddV2while/lstm_cell_28/mul:z:0while/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/add_1
while/lstm_cell_28/Sigmoid_2Sigmoid!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_2
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Relu_1¹
while/lstm_cell_28/mul_2Mul while/lstm_cell_28/Sigmoid_2:y:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_28/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_28/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_28/BiasAdd/ReadVariableOp)^while/lstm_cell_28/MatMul/ReadVariableOp+^while/lstm_cell_28/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_28_biasadd_readvariableop_resource4while_lstm_cell_28_biasadd_readvariableop_resource_0"l
3while_lstm_cell_28_matmul_1_readvariableop_resource5while_lstm_cell_28_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_28_matmul_readvariableop_resource3while_lstm_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_28/BiasAdd/ReadVariableOp)while/lstm_cell_28/BiasAdd/ReadVariableOp2T
(while/lstm_cell_28/MatMul/ReadVariableOp(while/lstm_cell_28/MatMul/ReadVariableOp2X
*while/lstm_cell_28/MatMul_1/ReadVariableOp*while/lstm_cell_28/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
ÿ?

D__inference_lstm_27_layer_call_and_return_conditional_losses_1058591

inputs(
lstm_cell_27_1058509:
ª¨(
lstm_cell_27_1058511:
ª¨#
lstm_cell_27_1058513:	¨
identity¢$lstm_cell_27/StatefulPartitionedCall¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¤
$lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_27_1058509lstm_cell_27_1058511lstm_cell_27_1058513*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_10584522&
$lstm_cell_27/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_27_1058509lstm_cell_27_1058511lstm_cell_27_1058513*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1058522*
condR
while_cond_1058521*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

Identity}
NoOpNoOp%^lstm_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 2L
$lstm_cell_27/StatefulPartitionedCall$lstm_cell_27/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
÷%
î
while_body_1058320
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_27_1058344_0:
ª¨0
while_lstm_cell_27_1058346_0:
ª¨+
while_lstm_cell_27_1058348_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_27_1058344:
ª¨.
while_lstm_cell_27_1058346:
ª¨)
while_lstm_cell_27_1058348:	¨¢*while/lstm_cell_27/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemè
*while/lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_27_1058344_0while_lstm_cell_27_1058346_0while_lstm_cell_27_1058348_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_10583062,
*while/lstm_cell_27/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_27/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_27/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_27/StatefulPartitionedCall*"
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
while_lstm_cell_27_1058344while_lstm_cell_27_1058344_0":
while_lstm_cell_27_1058346while_lstm_cell_27_1058346_0":
while_lstm_cell_27_1058348while_lstm_cell_27_1058348_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2X
*while/lstm_cell_27/StatefulPartitionedCall*while/lstm_cell_27/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
·
¸
)__inference_lstm_28_layer_call_fn_1063255

inputs
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_28_layer_call_and_return_conditional_losses_10602452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs


I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_1058904

inputs

states
states_12
matmul_readvariableop_resource:
ª¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates


I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_1064222

inputs
states_0
states_12
matmul_readvariableop_resource:
ª¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1
Þ
È
while_cond_1058521
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1058521___redundant_placeholder05
1while_while_cond_1058521___redundant_placeholder15
1while_while_cond_1058521___redundant_placeholder25
1while_while_cond_1058521___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
V
 
D__inference_lstm_28_layer_call_and_return_conditional_losses_1063398
inputs_0?
+lstm_cell_28_matmul_readvariableop_resource:
ª¨A
-lstm_cell_28_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_28_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_28/BiasAdd/ReadVariableOp¢"lstm_cell_28/MatMul/ReadVariableOp¢$lstm_cell_28/MatMul_1/ReadVariableOp¢whileF
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_28/MatMul/ReadVariableOpReadVariableOp+lstm_cell_28_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_28/MatMul/ReadVariableOp­
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0*lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul¼
$lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_28_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_28/MatMul_1/ReadVariableOp©
lstm_cell_28/MatMul_1MatMulzeros:output:0,lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul_1 
lstm_cell_28/addAddV2lstm_cell_28/MatMul:product:0lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/add´
#lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_28_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_28/BiasAdd/ReadVariableOp­
lstm_cell_28/BiasAddBiasAddlstm_cell_28/add:z:0+lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/BiasAdd~
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_28/split/split_dim÷
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_28/split
lstm_cell_28/SigmoidSigmoidlstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid
lstm_cell_28/Sigmoid_1Sigmoidlstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_1
lstm_cell_28/mulMullstm_cell_28/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul~
lstm_cell_28/ReluRelulstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu
lstm_cell_28/mul_1Mullstm_cell_28/Sigmoid:y:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_1
lstm_cell_28/add_1AddV2lstm_cell_28/mul:z:0lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/add_1
lstm_cell_28/Sigmoid_2Sigmoidlstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_2}
lstm_cell_28/Relu_1Relulstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu_1¡
lstm_cell_28/mul_2Mullstm_cell_28/Sigmoid_2:y:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_28_matmul_readvariableop_resource-lstm_cell_28_matmul_1_readvariableop_resource,lstm_cell_28_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1063314*
condR
while_cond_1063313*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_28/BiasAdd/ReadVariableOp#^lstm_cell_28/MatMul/ReadVariableOp%^lstm_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_28/BiasAdd/ReadVariableOp#lstm_cell_28/BiasAdd/ReadVariableOp2H
"lstm_cell_28/MatMul/ReadVariableOp"lstm_cell_28/MatMul/ReadVariableOp2L
$lstm_cell_28/MatMul_1/ReadVariableOp$lstm_cell_28/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
inputs/0


*__inference_dense_30_layer_call_fn_1063930

inputs
unknown:	ª
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
E__inference_dense_30_layer_call_and_return_conditional_losses_10599872
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
:ÿÿÿÿÿÿÿÿÿª: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
¹
e
,__inference_dropout_43_layer_call_fn_1062551

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
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_10604622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Ö!
ÿ
E__inference_dense_29_layer_call_and_return_conditional_losses_1063894

inputs5
!tensordot_readvariableop_resource:
ªª.
biasadd_readvariableop_resource:	ª
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ªª*
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
:ÿÿÿÿÿÿÿÿÿª2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ª2
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
:ÿÿÿÿÿÿÿÿÿª2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ª*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿª: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
³?
Õ
while_body_1063600
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_28_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_28_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_28_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_28_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_28_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_28/BiasAdd/ReadVariableOp¢(while/lstm_cell_28/MatMul/ReadVariableOp¢*while/lstm_cell_28/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_28/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_28_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_28/MatMul/ReadVariableOp×
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMulÐ
*while/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_28_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_28/MatMul_1/ReadVariableOpÀ
while/lstm_cell_28/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMul_1¸
while/lstm_cell_28/addAddV2#while/lstm_cell_28/MatMul:product:0%while/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/addÈ
)while/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_28_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_28/BiasAdd/ReadVariableOpÅ
while/lstm_cell_28/BiasAddBiasAddwhile/lstm_cell_28/add:z:01while/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/BiasAdd
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_28/split/split_dim
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0#while/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_28/split
while/lstm_cell_28/SigmoidSigmoid!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid
while/lstm_cell_28/Sigmoid_1Sigmoid!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_1¡
while/lstm_cell_28/mulMul while/lstm_cell_28/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul
while/lstm_cell_28/ReluRelu!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Reluµ
while/lstm_cell_28/mul_1Mulwhile/lstm_cell_28/Sigmoid:y:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_1ª
while/lstm_cell_28/add_1AddV2while/lstm_cell_28/mul:z:0while/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/add_1
while/lstm_cell_28/Sigmoid_2Sigmoid!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_2
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Relu_1¹
while/lstm_cell_28/mul_2Mul while/lstm_cell_28/Sigmoid_2:y:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_28/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_28/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_28/BiasAdd/ReadVariableOp)^while/lstm_cell_28/MatMul/ReadVariableOp+^while/lstm_cell_28/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_28_biasadd_readvariableop_resource4while_lstm_cell_28_biasadd_readvariableop_resource_0"l
3while_lstm_cell_28_matmul_1_readvariableop_resource5while_lstm_cell_28_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_28_matmul_readvariableop_resource3while_lstm_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_28/BiasAdd/ReadVariableOp)while/lstm_cell_28/BiasAdd/ReadVariableOp2T
(while/lstm_cell_28/MatMul/ReadVariableOp(while/lstm_cell_28/MatMul/ReadVariableOp2X
*while/lstm_cell_28/MatMul_1/ReadVariableOp*while/lstm_cell_28/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1062170
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1062170___redundant_placeholder05
1while_while_cond_1062170___redundant_placeholder15
1while_while_cond_1062170___redundant_placeholder25
1while_while_cond_1062170___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
Ï

è
lstm_28_while_cond_1061279,
(lstm_28_while_lstm_28_while_loop_counter2
.lstm_28_while_lstm_28_while_maximum_iterations
lstm_28_while_placeholder
lstm_28_while_placeholder_1
lstm_28_while_placeholder_2
lstm_28_while_placeholder_3.
*lstm_28_while_less_lstm_28_strided_slice_1E
Alstm_28_while_lstm_28_while_cond_1061279___redundant_placeholder0E
Alstm_28_while_lstm_28_while_cond_1061279___redundant_placeholder1E
Alstm_28_while_lstm_28_while_cond_1061279___redundant_placeholder2E
Alstm_28_while_lstm_28_while_cond_1061279___redundant_placeholder3
lstm_28_while_identity

lstm_28/while/LessLesslstm_28_while_placeholder*lstm_28_while_less_lstm_28_strided_slice_1*
T0*
_output_shapes
: 2
lstm_28/while/Lessu
lstm_28/while/IdentityIdentitylstm_28/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_28/while/Identity"9
lstm_28_while_identitylstm_28/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
È
ù
.__inference_lstm_cell_27_layer_call_fn_1064075

inputs
states_0
states_1
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
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
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_10583062
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1
Íf
ï
 __inference__traced_save_1064421
file_prefix.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_26_lstm_cell_26_kernel_read_readvariableopD
@savev2_lstm_26_lstm_cell_26_recurrent_kernel_read_readvariableop8
4savev2_lstm_26_lstm_cell_26_bias_read_readvariableop:
6savev2_lstm_27_lstm_cell_27_kernel_read_readvariableopD
@savev2_lstm_27_lstm_cell_27_recurrent_kernel_read_readvariableop8
4savev2_lstm_27_lstm_cell_27_bias_read_readvariableop:
6savev2_lstm_28_lstm_cell_28_kernel_read_readvariableopD
@savev2_lstm_28_lstm_cell_28_recurrent_kernel_read_readvariableop8
4savev2_lstm_28_lstm_cell_28_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableop5
1savev2_adam_dense_30_kernel_m_read_readvariableop3
/savev2_adam_dense_30_bias_m_read_readvariableopA
=savev2_adam_lstm_26_lstm_cell_26_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_26_lstm_cell_26_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_26_lstm_cell_26_bias_m_read_readvariableopA
=savev2_adam_lstm_27_lstm_cell_27_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_27_lstm_cell_27_bias_m_read_readvariableopA
=savev2_adam_lstm_28_lstm_cell_28_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_28_lstm_cell_28_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_28_lstm_cell_28_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableop5
1savev2_adam_dense_30_kernel_v_read_readvariableop3
/savev2_adam_dense_30_bias_v_read_readvariableopA
=savev2_adam_lstm_26_lstm_cell_26_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_26_lstm_cell_26_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_26_lstm_cell_26_bias_v_read_readvariableopA
=savev2_adam_lstm_27_lstm_cell_27_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_27_lstm_cell_27_bias_v_read_readvariableopA
=savev2_adam_lstm_28_lstm_cell_28_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_28_lstm_cell_28_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_28_lstm_cell_28_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_26_lstm_cell_26_kernel_read_readvariableop@savev2_lstm_26_lstm_cell_26_recurrent_kernel_read_readvariableop4savev2_lstm_26_lstm_cell_26_bias_read_readvariableop6savev2_lstm_27_lstm_cell_27_kernel_read_readvariableop@savev2_lstm_27_lstm_cell_27_recurrent_kernel_read_readvariableop4savev2_lstm_27_lstm_cell_27_bias_read_readvariableop6savev2_lstm_28_lstm_cell_28_kernel_read_readvariableop@savev2_lstm_28_lstm_cell_28_recurrent_kernel_read_readvariableop4savev2_lstm_28_lstm_cell_28_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableop1savev2_adam_dense_30_kernel_m_read_readvariableop/savev2_adam_dense_30_bias_m_read_readvariableop=savev2_adam_lstm_26_lstm_cell_26_kernel_m_read_readvariableopGsavev2_adam_lstm_26_lstm_cell_26_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_26_lstm_cell_26_bias_m_read_readvariableop=savev2_adam_lstm_27_lstm_cell_27_kernel_m_read_readvariableopGsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_27_lstm_cell_27_bias_m_read_readvariableop=savev2_adam_lstm_28_lstm_cell_28_kernel_m_read_readvariableopGsavev2_adam_lstm_28_lstm_cell_28_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_28_lstm_cell_28_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableop1savev2_adam_dense_30_kernel_v_read_readvariableop/savev2_adam_dense_30_bias_v_read_readvariableop=savev2_adam_lstm_26_lstm_cell_26_kernel_v_read_readvariableopGsavev2_adam_lstm_26_lstm_cell_26_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_26_lstm_cell_26_bias_v_read_readvariableop=savev2_adam_lstm_27_lstm_cell_27_kernel_v_read_readvariableopGsavev2_adam_lstm_27_lstm_cell_27_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_27_lstm_cell_27_bias_v_read_readvariableop=savev2_adam_lstm_28_lstm_cell_28_kernel_v_read_readvariableopGsavev2_adam_lstm_28_lstm_cell_28_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_28_lstm_cell_28_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
ªª:ª:	ª:: : : : : :	¨:
ª¨:¨:
ª¨:
ª¨:¨:
ª¨:
ª¨:¨: : : : :
ªª:ª:	ª::	¨:
ª¨:¨:
ª¨:
ª¨:¨:
ª¨:
ª¨:¨:
ªª:ª:	ª::	¨:
ª¨:¨:
ª¨:
ª¨:¨:
ª¨:
ª¨:¨: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ªª:!

_output_shapes	
:ª:%!

_output_shapes
:	ª: 
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
:	¨:&"
 
_output_shapes
:
ª¨:!

_output_shapes	
:¨:&"
 
_output_shapes
:
ª¨:&"
 
_output_shapes
:
ª¨:!

_output_shapes	
:¨:&"
 
_output_shapes
:
ª¨:&"
 
_output_shapes
:
ª¨:!

_output_shapes	
:¨:
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
ªª:!

_output_shapes	
:ª:%!

_output_shapes
:	ª: 

_output_shapes
::%!

_output_shapes
:	¨:&"
 
_output_shapes
:
ª¨:!

_output_shapes	
:¨:&"
 
_output_shapes
:
ª¨:&"
 
_output_shapes
:
ª¨:! 

_output_shapes	
:¨:&!"
 
_output_shapes
:
ª¨:&""
 
_output_shapes
:
ª¨:!#

_output_shapes	
:¨:&$"
 
_output_shapes
:
ªª:!%

_output_shapes	
:ª:%&!

_output_shapes
:	ª: '

_output_shapes
::%(!

_output_shapes
:	¨:&)"
 
_output_shapes
:
ª¨:!*

_output_shapes	
:¨:&+"
 
_output_shapes
:
ª¨:&,"
 
_output_shapes
:
ª¨:!-

_output_shapes	
:¨:&."
 
_output_shapes
:
ª¨:&/"
 
_output_shapes
:
ª¨:!0

_output_shapes	
:¨:1

_output_shapes
: 
Þ
È
while_cond_1060348
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1060348___redundant_placeholder05
1while_while_cond_1060348___redundant_placeholder15
1while_while_cond_1060348___redundant_placeholder25
1while_while_cond_1060348___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
±Ë
²
"__inference__wrapped_model_1057641
lstm_26_inputT
Asequential_12_lstm_26_lstm_cell_26_matmul_readvariableop_resource:	¨W
Csequential_12_lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨Q
Bsequential_12_lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	¨U
Asequential_12_lstm_27_lstm_cell_27_matmul_readvariableop_resource:
ª¨W
Csequential_12_lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨Q
Bsequential_12_lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	¨U
Asequential_12_lstm_28_lstm_cell_28_matmul_readvariableop_resource:
ª¨W
Csequential_12_lstm_28_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨Q
Bsequential_12_lstm_28_lstm_cell_28_biasadd_readvariableop_resource:	¨L
8sequential_12_dense_29_tensordot_readvariableop_resource:
ªªE
6sequential_12_dense_29_biasadd_readvariableop_resource:	ªK
8sequential_12_dense_30_tensordot_readvariableop_resource:	ªD
6sequential_12_dense_30_biasadd_readvariableop_resource:
identity¢-sequential_12/dense_29/BiasAdd/ReadVariableOp¢/sequential_12/dense_29/Tensordot/ReadVariableOp¢-sequential_12/dense_30/BiasAdd/ReadVariableOp¢/sequential_12/dense_30/Tensordot/ReadVariableOp¢9sequential_12/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp¢8sequential_12/lstm_26/lstm_cell_26/MatMul/ReadVariableOp¢:sequential_12/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp¢sequential_12/lstm_26/while¢9sequential_12/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp¢8sequential_12/lstm_27/lstm_cell_27/MatMul/ReadVariableOp¢:sequential_12/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp¢sequential_12/lstm_27/while¢9sequential_12/lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp¢8sequential_12/lstm_28/lstm_cell_28/MatMul/ReadVariableOp¢:sequential_12/lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp¢sequential_12/lstm_28/whilew
sequential_12/lstm_26/ShapeShapelstm_26_input*
T0*
_output_shapes
:2
sequential_12/lstm_26/Shape 
)sequential_12/lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_12/lstm_26/strided_slice/stack¤
+sequential_12/lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_26/strided_slice/stack_1¤
+sequential_12/lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_26/strided_slice/stack_2æ
#sequential_12/lstm_26/strided_sliceStridedSlice$sequential_12/lstm_26/Shape:output:02sequential_12/lstm_26/strided_slice/stack:output:04sequential_12/lstm_26/strided_slice/stack_1:output:04sequential_12/lstm_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_12/lstm_26/strided_slice
$sequential_12/lstm_26/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2&
$sequential_12/lstm_26/zeros/packed/1Û
"sequential_12/lstm_26/zeros/packedPack,sequential_12/lstm_26/strided_slice:output:0-sequential_12/lstm_26/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_12/lstm_26/zeros/packed
!sequential_12/lstm_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_12/lstm_26/zeros/ConstÎ
sequential_12/lstm_26/zerosFill+sequential_12/lstm_26/zeros/packed:output:0*sequential_12/lstm_26/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
sequential_12/lstm_26/zeros
&sequential_12/lstm_26/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2(
&sequential_12/lstm_26/zeros_1/packed/1á
$sequential_12/lstm_26/zeros_1/packedPack,sequential_12/lstm_26/strided_slice:output:0/sequential_12/lstm_26/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_12/lstm_26/zeros_1/packed
#sequential_12/lstm_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_12/lstm_26/zeros_1/ConstÖ
sequential_12/lstm_26/zeros_1Fill-sequential_12/lstm_26/zeros_1/packed:output:0,sequential_12/lstm_26/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
sequential_12/lstm_26/zeros_1¡
$sequential_12/lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_12/lstm_26/transpose/permÃ
sequential_12/lstm_26/transpose	Transposelstm_26_input-sequential_12/lstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_12/lstm_26/transpose
sequential_12/lstm_26/Shape_1Shape#sequential_12/lstm_26/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/lstm_26/Shape_1¤
+sequential_12/lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_26/strided_slice_1/stack¨
-sequential_12/lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_26/strided_slice_1/stack_1¨
-sequential_12/lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_26/strided_slice_1/stack_2ò
%sequential_12/lstm_26/strided_slice_1StridedSlice&sequential_12/lstm_26/Shape_1:output:04sequential_12/lstm_26/strided_slice_1/stack:output:06sequential_12/lstm_26/strided_slice_1/stack_1:output:06sequential_12/lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_12/lstm_26/strided_slice_1±
1sequential_12/lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_12/lstm_26/TensorArrayV2/element_shape
#sequential_12/lstm_26/TensorArrayV2TensorListReserve:sequential_12/lstm_26/TensorArrayV2/element_shape:output:0.sequential_12/lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_12/lstm_26/TensorArrayV2ë
Ksequential_12/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2M
Ksequential_12/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_12/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_12/lstm_26/transpose:y:0Tsequential_12/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_12/lstm_26/TensorArrayUnstack/TensorListFromTensor¤
+sequential_12/lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_26/strided_slice_2/stack¨
-sequential_12/lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_26/strided_slice_2/stack_1¨
-sequential_12/lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_26/strided_slice_2/stack_2
%sequential_12/lstm_26/strided_slice_2StridedSlice#sequential_12/lstm_26/transpose:y:04sequential_12/lstm_26/strided_slice_2/stack:output:06sequential_12/lstm_26/strided_slice_2/stack_1:output:06sequential_12/lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_12/lstm_26/strided_slice_2÷
8sequential_12/lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOpAsequential_12_lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02:
8sequential_12/lstm_26/lstm_cell_26/MatMul/ReadVariableOp
)sequential_12/lstm_26/lstm_cell_26/MatMulMatMul.sequential_12/lstm_26/strided_slice_2:output:0@sequential_12/lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2+
)sequential_12/lstm_26/lstm_cell_26/MatMulþ
:sequential_12/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_26_lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02<
:sequential_12/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp
+sequential_12/lstm_26/lstm_cell_26/MatMul_1MatMul$sequential_12/lstm_26/zeros:output:0Bsequential_12/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2-
+sequential_12/lstm_26/lstm_cell_26/MatMul_1ø
&sequential_12/lstm_26/lstm_cell_26/addAddV23sequential_12/lstm_26/lstm_cell_26/MatMul:product:05sequential_12/lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2(
&sequential_12/lstm_26/lstm_cell_26/addö
9sequential_12/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02;
9sequential_12/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp
*sequential_12/lstm_26/lstm_cell_26/BiasAddBiasAdd*sequential_12/lstm_26/lstm_cell_26/add:z:0Asequential_12/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2,
*sequential_12/lstm_26/lstm_cell_26/BiasAddª
2sequential_12/lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_26/lstm_cell_26/split/split_dimÏ
(sequential_12/lstm_26/lstm_cell_26/splitSplit;sequential_12/lstm_26/lstm_cell_26/split/split_dim:output:03sequential_12/lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2*
(sequential_12/lstm_26/lstm_cell_26/splitÉ
*sequential_12/lstm_26/lstm_cell_26/SigmoidSigmoid1sequential_12/lstm_26/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2,
*sequential_12/lstm_26/lstm_cell_26/SigmoidÍ
,sequential_12/lstm_26/lstm_cell_26/Sigmoid_1Sigmoid1sequential_12/lstm_26/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2.
,sequential_12/lstm_26/lstm_cell_26/Sigmoid_1ä
&sequential_12/lstm_26/lstm_cell_26/mulMul0sequential_12/lstm_26/lstm_cell_26/Sigmoid_1:y:0&sequential_12/lstm_26/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2(
&sequential_12/lstm_26/lstm_cell_26/mulÀ
'sequential_12/lstm_26/lstm_cell_26/ReluRelu1sequential_12/lstm_26/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2)
'sequential_12/lstm_26/lstm_cell_26/Reluõ
(sequential_12/lstm_26/lstm_cell_26/mul_1Mul.sequential_12/lstm_26/lstm_cell_26/Sigmoid:y:05sequential_12/lstm_26/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2*
(sequential_12/lstm_26/lstm_cell_26/mul_1ê
(sequential_12/lstm_26/lstm_cell_26/add_1AddV2*sequential_12/lstm_26/lstm_cell_26/mul:z:0,sequential_12/lstm_26/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2*
(sequential_12/lstm_26/lstm_cell_26/add_1Í
,sequential_12/lstm_26/lstm_cell_26/Sigmoid_2Sigmoid1sequential_12/lstm_26/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2.
,sequential_12/lstm_26/lstm_cell_26/Sigmoid_2¿
)sequential_12/lstm_26/lstm_cell_26/Relu_1Relu,sequential_12/lstm_26/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2+
)sequential_12/lstm_26/lstm_cell_26/Relu_1ù
(sequential_12/lstm_26/lstm_cell_26/mul_2Mul0sequential_12/lstm_26/lstm_cell_26/Sigmoid_2:y:07sequential_12/lstm_26/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2*
(sequential_12/lstm_26/lstm_cell_26/mul_2»
3sequential_12/lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   25
3sequential_12/lstm_26/TensorArrayV2_1/element_shape
%sequential_12/lstm_26/TensorArrayV2_1TensorListReserve<sequential_12/lstm_26/TensorArrayV2_1/element_shape:output:0.sequential_12/lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_12/lstm_26/TensorArrayV2_1z
sequential_12/lstm_26/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/lstm_26/time«
.sequential_12/lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_12/lstm_26/while/maximum_iterations
(sequential_12/lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_12/lstm_26/while/loop_counterÛ
sequential_12/lstm_26/whileWhile1sequential_12/lstm_26/while/loop_counter:output:07sequential_12/lstm_26/while/maximum_iterations:output:0#sequential_12/lstm_26/time:output:0.sequential_12/lstm_26/TensorArrayV2_1:handle:0$sequential_12/lstm_26/zeros:output:0&sequential_12/lstm_26/zeros_1:output:0.sequential_12/lstm_26/strided_slice_1:output:0Msequential_12/lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_26_lstm_cell_26_matmul_readvariableop_resourceCsequential_12_lstm_26_lstm_cell_26_matmul_1_readvariableop_resourceBsequential_12_lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_12_lstm_26_while_body_1057222*4
cond,R*
(sequential_12_lstm_26_while_cond_1057221*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
sequential_12/lstm_26/whileá
Fsequential_12/lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2H
Fsequential_12/lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_12/lstm_26/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_26/while:output:3Osequential_12/lstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02:
8sequential_12/lstm_26/TensorArrayV2Stack/TensorListStack­
+sequential_12/lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_12/lstm_26/strided_slice_3/stack¨
-sequential_12/lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_12/lstm_26/strided_slice_3/stack_1¨
-sequential_12/lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_26/strided_slice_3/stack_2
%sequential_12/lstm_26/strided_slice_3StridedSliceAsequential_12/lstm_26/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_26/strided_slice_3/stack:output:06sequential_12/lstm_26/strided_slice_3/stack_1:output:06sequential_12/lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2'
%sequential_12/lstm_26/strided_slice_3¥
&sequential_12/lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_26/transpose_1/permþ
!sequential_12/lstm_26/transpose_1	TransposeAsequential_12/lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_26/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!sequential_12/lstm_26/transpose_1
sequential_12/lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_26/runtime°
!sequential_12/dropout_43/IdentityIdentity%sequential_12/lstm_26/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!sequential_12/dropout_43/Identity
sequential_12/lstm_27/ShapeShape*sequential_12/dropout_43/Identity:output:0*
T0*
_output_shapes
:2
sequential_12/lstm_27/Shape 
)sequential_12/lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_12/lstm_27/strided_slice/stack¤
+sequential_12/lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_27/strided_slice/stack_1¤
+sequential_12/lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_27/strided_slice/stack_2æ
#sequential_12/lstm_27/strided_sliceStridedSlice$sequential_12/lstm_27/Shape:output:02sequential_12/lstm_27/strided_slice/stack:output:04sequential_12/lstm_27/strided_slice/stack_1:output:04sequential_12/lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_12/lstm_27/strided_slice
$sequential_12/lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2&
$sequential_12/lstm_27/zeros/packed/1Û
"sequential_12/lstm_27/zeros/packedPack,sequential_12/lstm_27/strided_slice:output:0-sequential_12/lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_12/lstm_27/zeros/packed
!sequential_12/lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_12/lstm_27/zeros/ConstÎ
sequential_12/lstm_27/zerosFill+sequential_12/lstm_27/zeros/packed:output:0*sequential_12/lstm_27/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
sequential_12/lstm_27/zeros
&sequential_12/lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2(
&sequential_12/lstm_27/zeros_1/packed/1á
$sequential_12/lstm_27/zeros_1/packedPack,sequential_12/lstm_27/strided_slice:output:0/sequential_12/lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_12/lstm_27/zeros_1/packed
#sequential_12/lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_12/lstm_27/zeros_1/ConstÖ
sequential_12/lstm_27/zeros_1Fill-sequential_12/lstm_27/zeros_1/packed:output:0,sequential_12/lstm_27/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
sequential_12/lstm_27/zeros_1¡
$sequential_12/lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_12/lstm_27/transpose/permá
sequential_12/lstm_27/transpose	Transpose*sequential_12/dropout_43/Identity:output:0-sequential_12/lstm_27/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
sequential_12/lstm_27/transpose
sequential_12/lstm_27/Shape_1Shape#sequential_12/lstm_27/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/lstm_27/Shape_1¤
+sequential_12/lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_27/strided_slice_1/stack¨
-sequential_12/lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_27/strided_slice_1/stack_1¨
-sequential_12/lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_27/strided_slice_1/stack_2ò
%sequential_12/lstm_27/strided_slice_1StridedSlice&sequential_12/lstm_27/Shape_1:output:04sequential_12/lstm_27/strided_slice_1/stack:output:06sequential_12/lstm_27/strided_slice_1/stack_1:output:06sequential_12/lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_12/lstm_27/strided_slice_1±
1sequential_12/lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_12/lstm_27/TensorArrayV2/element_shape
#sequential_12/lstm_27/TensorArrayV2TensorListReserve:sequential_12/lstm_27/TensorArrayV2/element_shape:output:0.sequential_12/lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_12/lstm_27/TensorArrayV2ë
Ksequential_12/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2M
Ksequential_12/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_12/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_12/lstm_27/transpose:y:0Tsequential_12/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_12/lstm_27/TensorArrayUnstack/TensorListFromTensor¤
+sequential_12/lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_27/strided_slice_2/stack¨
-sequential_12/lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_27/strided_slice_2/stack_1¨
-sequential_12/lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_27/strided_slice_2/stack_2
%sequential_12/lstm_27/strided_slice_2StridedSlice#sequential_12/lstm_27/transpose:y:04sequential_12/lstm_27/strided_slice_2/stack:output:06sequential_12/lstm_27/strided_slice_2/stack_1:output:06sequential_12/lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2'
%sequential_12/lstm_27/strided_slice_2ø
8sequential_12/lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOpAsequential_12_lstm_27_lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02:
8sequential_12/lstm_27/lstm_cell_27/MatMul/ReadVariableOp
)sequential_12/lstm_27/lstm_cell_27/MatMulMatMul.sequential_12/lstm_27/strided_slice_2:output:0@sequential_12/lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2+
)sequential_12/lstm_27/lstm_cell_27/MatMulþ
:sequential_12/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_27_lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02<
:sequential_12/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp
+sequential_12/lstm_27/lstm_cell_27/MatMul_1MatMul$sequential_12/lstm_27/zeros:output:0Bsequential_12/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2-
+sequential_12/lstm_27/lstm_cell_27/MatMul_1ø
&sequential_12/lstm_27/lstm_cell_27/addAddV23sequential_12/lstm_27/lstm_cell_27/MatMul:product:05sequential_12/lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2(
&sequential_12/lstm_27/lstm_cell_27/addö
9sequential_12/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02;
9sequential_12/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp
*sequential_12/lstm_27/lstm_cell_27/BiasAddBiasAdd*sequential_12/lstm_27/lstm_cell_27/add:z:0Asequential_12/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2,
*sequential_12/lstm_27/lstm_cell_27/BiasAddª
2sequential_12/lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_27/lstm_cell_27/split/split_dimÏ
(sequential_12/lstm_27/lstm_cell_27/splitSplit;sequential_12/lstm_27/lstm_cell_27/split/split_dim:output:03sequential_12/lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2*
(sequential_12/lstm_27/lstm_cell_27/splitÉ
*sequential_12/lstm_27/lstm_cell_27/SigmoidSigmoid1sequential_12/lstm_27/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2,
*sequential_12/lstm_27/lstm_cell_27/SigmoidÍ
,sequential_12/lstm_27/lstm_cell_27/Sigmoid_1Sigmoid1sequential_12/lstm_27/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2.
,sequential_12/lstm_27/lstm_cell_27/Sigmoid_1ä
&sequential_12/lstm_27/lstm_cell_27/mulMul0sequential_12/lstm_27/lstm_cell_27/Sigmoid_1:y:0&sequential_12/lstm_27/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2(
&sequential_12/lstm_27/lstm_cell_27/mulÀ
'sequential_12/lstm_27/lstm_cell_27/ReluRelu1sequential_12/lstm_27/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2)
'sequential_12/lstm_27/lstm_cell_27/Reluõ
(sequential_12/lstm_27/lstm_cell_27/mul_1Mul.sequential_12/lstm_27/lstm_cell_27/Sigmoid:y:05sequential_12/lstm_27/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2*
(sequential_12/lstm_27/lstm_cell_27/mul_1ê
(sequential_12/lstm_27/lstm_cell_27/add_1AddV2*sequential_12/lstm_27/lstm_cell_27/mul:z:0,sequential_12/lstm_27/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2*
(sequential_12/lstm_27/lstm_cell_27/add_1Í
,sequential_12/lstm_27/lstm_cell_27/Sigmoid_2Sigmoid1sequential_12/lstm_27/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2.
,sequential_12/lstm_27/lstm_cell_27/Sigmoid_2¿
)sequential_12/lstm_27/lstm_cell_27/Relu_1Relu,sequential_12/lstm_27/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2+
)sequential_12/lstm_27/lstm_cell_27/Relu_1ù
(sequential_12/lstm_27/lstm_cell_27/mul_2Mul0sequential_12/lstm_27/lstm_cell_27/Sigmoid_2:y:07sequential_12/lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2*
(sequential_12/lstm_27/lstm_cell_27/mul_2»
3sequential_12/lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   25
3sequential_12/lstm_27/TensorArrayV2_1/element_shape
%sequential_12/lstm_27/TensorArrayV2_1TensorListReserve<sequential_12/lstm_27/TensorArrayV2_1/element_shape:output:0.sequential_12/lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_12/lstm_27/TensorArrayV2_1z
sequential_12/lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/lstm_27/time«
.sequential_12/lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_12/lstm_27/while/maximum_iterations
(sequential_12/lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_12/lstm_27/while/loop_counterÛ
sequential_12/lstm_27/whileWhile1sequential_12/lstm_27/while/loop_counter:output:07sequential_12/lstm_27/while/maximum_iterations:output:0#sequential_12/lstm_27/time:output:0.sequential_12/lstm_27/TensorArrayV2_1:handle:0$sequential_12/lstm_27/zeros:output:0&sequential_12/lstm_27/zeros_1:output:0.sequential_12/lstm_27/strided_slice_1:output:0Msequential_12/lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_27_lstm_cell_27_matmul_readvariableop_resourceCsequential_12_lstm_27_lstm_cell_27_matmul_1_readvariableop_resourceBsequential_12_lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_12_lstm_27_while_body_1057362*4
cond,R*
(sequential_12_lstm_27_while_cond_1057361*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
sequential_12/lstm_27/whileá
Fsequential_12/lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2H
Fsequential_12/lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_12/lstm_27/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_27/while:output:3Osequential_12/lstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02:
8sequential_12/lstm_27/TensorArrayV2Stack/TensorListStack­
+sequential_12/lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_12/lstm_27/strided_slice_3/stack¨
-sequential_12/lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_12/lstm_27/strided_slice_3/stack_1¨
-sequential_12/lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_27/strided_slice_3/stack_2
%sequential_12/lstm_27/strided_slice_3StridedSliceAsequential_12/lstm_27/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_27/strided_slice_3/stack:output:06sequential_12/lstm_27/strided_slice_3/stack_1:output:06sequential_12/lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2'
%sequential_12/lstm_27/strided_slice_3¥
&sequential_12/lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_27/transpose_1/permþ
!sequential_12/lstm_27/transpose_1	TransposeAsequential_12/lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_27/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!sequential_12/lstm_27/transpose_1
sequential_12/lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_27/runtime°
!sequential_12/dropout_44/IdentityIdentity%sequential_12/lstm_27/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!sequential_12/dropout_44/Identity
sequential_12/lstm_28/ShapeShape*sequential_12/dropout_44/Identity:output:0*
T0*
_output_shapes
:2
sequential_12/lstm_28/Shape 
)sequential_12/lstm_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_12/lstm_28/strided_slice/stack¤
+sequential_12/lstm_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_28/strided_slice/stack_1¤
+sequential_12/lstm_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_28/strided_slice/stack_2æ
#sequential_12/lstm_28/strided_sliceStridedSlice$sequential_12/lstm_28/Shape:output:02sequential_12/lstm_28/strided_slice/stack:output:04sequential_12/lstm_28/strided_slice/stack_1:output:04sequential_12/lstm_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_12/lstm_28/strided_slice
$sequential_12/lstm_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2&
$sequential_12/lstm_28/zeros/packed/1Û
"sequential_12/lstm_28/zeros/packedPack,sequential_12/lstm_28/strided_slice:output:0-sequential_12/lstm_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_12/lstm_28/zeros/packed
!sequential_12/lstm_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_12/lstm_28/zeros/ConstÎ
sequential_12/lstm_28/zerosFill+sequential_12/lstm_28/zeros/packed:output:0*sequential_12/lstm_28/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
sequential_12/lstm_28/zeros
&sequential_12/lstm_28/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2(
&sequential_12/lstm_28/zeros_1/packed/1á
$sequential_12/lstm_28/zeros_1/packedPack,sequential_12/lstm_28/strided_slice:output:0/sequential_12/lstm_28/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_12/lstm_28/zeros_1/packed
#sequential_12/lstm_28/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_12/lstm_28/zeros_1/ConstÖ
sequential_12/lstm_28/zeros_1Fill-sequential_12/lstm_28/zeros_1/packed:output:0,sequential_12/lstm_28/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
sequential_12/lstm_28/zeros_1¡
$sequential_12/lstm_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_12/lstm_28/transpose/permá
sequential_12/lstm_28/transpose	Transpose*sequential_12/dropout_44/Identity:output:0-sequential_12/lstm_28/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
sequential_12/lstm_28/transpose
sequential_12/lstm_28/Shape_1Shape#sequential_12/lstm_28/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/lstm_28/Shape_1¤
+sequential_12/lstm_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_28/strided_slice_1/stack¨
-sequential_12/lstm_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_28/strided_slice_1/stack_1¨
-sequential_12/lstm_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_28/strided_slice_1/stack_2ò
%sequential_12/lstm_28/strided_slice_1StridedSlice&sequential_12/lstm_28/Shape_1:output:04sequential_12/lstm_28/strided_slice_1/stack:output:06sequential_12/lstm_28/strided_slice_1/stack_1:output:06sequential_12/lstm_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_12/lstm_28/strided_slice_1±
1sequential_12/lstm_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_12/lstm_28/TensorArrayV2/element_shape
#sequential_12/lstm_28/TensorArrayV2TensorListReserve:sequential_12/lstm_28/TensorArrayV2/element_shape:output:0.sequential_12/lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_12/lstm_28/TensorArrayV2ë
Ksequential_12/lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2M
Ksequential_12/lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_12/lstm_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_12/lstm_28/transpose:y:0Tsequential_12/lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_12/lstm_28/TensorArrayUnstack/TensorListFromTensor¤
+sequential_12/lstm_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_28/strided_slice_2/stack¨
-sequential_12/lstm_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_28/strided_slice_2/stack_1¨
-sequential_12/lstm_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_28/strided_slice_2/stack_2
%sequential_12/lstm_28/strided_slice_2StridedSlice#sequential_12/lstm_28/transpose:y:04sequential_12/lstm_28/strided_slice_2/stack:output:06sequential_12/lstm_28/strided_slice_2/stack_1:output:06sequential_12/lstm_28/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2'
%sequential_12/lstm_28/strided_slice_2ø
8sequential_12/lstm_28/lstm_cell_28/MatMul/ReadVariableOpReadVariableOpAsequential_12_lstm_28_lstm_cell_28_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02:
8sequential_12/lstm_28/lstm_cell_28/MatMul/ReadVariableOp
)sequential_12/lstm_28/lstm_cell_28/MatMulMatMul.sequential_12/lstm_28/strided_slice_2:output:0@sequential_12/lstm_28/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2+
)sequential_12/lstm_28/lstm_cell_28/MatMulþ
:sequential_12/lstm_28/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_28_lstm_cell_28_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02<
:sequential_12/lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp
+sequential_12/lstm_28/lstm_cell_28/MatMul_1MatMul$sequential_12/lstm_28/zeros:output:0Bsequential_12/lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2-
+sequential_12/lstm_28/lstm_cell_28/MatMul_1ø
&sequential_12/lstm_28/lstm_cell_28/addAddV23sequential_12/lstm_28/lstm_cell_28/MatMul:product:05sequential_12/lstm_28/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2(
&sequential_12/lstm_28/lstm_cell_28/addö
9sequential_12/lstm_28/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_28_lstm_cell_28_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02;
9sequential_12/lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp
*sequential_12/lstm_28/lstm_cell_28/BiasAddBiasAdd*sequential_12/lstm_28/lstm_cell_28/add:z:0Asequential_12/lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2,
*sequential_12/lstm_28/lstm_cell_28/BiasAddª
2sequential_12/lstm_28/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_28/lstm_cell_28/split/split_dimÏ
(sequential_12/lstm_28/lstm_cell_28/splitSplit;sequential_12/lstm_28/lstm_cell_28/split/split_dim:output:03sequential_12/lstm_28/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2*
(sequential_12/lstm_28/lstm_cell_28/splitÉ
*sequential_12/lstm_28/lstm_cell_28/SigmoidSigmoid1sequential_12/lstm_28/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2,
*sequential_12/lstm_28/lstm_cell_28/SigmoidÍ
,sequential_12/lstm_28/lstm_cell_28/Sigmoid_1Sigmoid1sequential_12/lstm_28/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2.
,sequential_12/lstm_28/lstm_cell_28/Sigmoid_1ä
&sequential_12/lstm_28/lstm_cell_28/mulMul0sequential_12/lstm_28/lstm_cell_28/Sigmoid_1:y:0&sequential_12/lstm_28/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2(
&sequential_12/lstm_28/lstm_cell_28/mulÀ
'sequential_12/lstm_28/lstm_cell_28/ReluRelu1sequential_12/lstm_28/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2)
'sequential_12/lstm_28/lstm_cell_28/Reluõ
(sequential_12/lstm_28/lstm_cell_28/mul_1Mul.sequential_12/lstm_28/lstm_cell_28/Sigmoid:y:05sequential_12/lstm_28/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2*
(sequential_12/lstm_28/lstm_cell_28/mul_1ê
(sequential_12/lstm_28/lstm_cell_28/add_1AddV2*sequential_12/lstm_28/lstm_cell_28/mul:z:0,sequential_12/lstm_28/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2*
(sequential_12/lstm_28/lstm_cell_28/add_1Í
,sequential_12/lstm_28/lstm_cell_28/Sigmoid_2Sigmoid1sequential_12/lstm_28/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2.
,sequential_12/lstm_28/lstm_cell_28/Sigmoid_2¿
)sequential_12/lstm_28/lstm_cell_28/Relu_1Relu,sequential_12/lstm_28/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2+
)sequential_12/lstm_28/lstm_cell_28/Relu_1ù
(sequential_12/lstm_28/lstm_cell_28/mul_2Mul0sequential_12/lstm_28/lstm_cell_28/Sigmoid_2:y:07sequential_12/lstm_28/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2*
(sequential_12/lstm_28/lstm_cell_28/mul_2»
3sequential_12/lstm_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   25
3sequential_12/lstm_28/TensorArrayV2_1/element_shape
%sequential_12/lstm_28/TensorArrayV2_1TensorListReserve<sequential_12/lstm_28/TensorArrayV2_1/element_shape:output:0.sequential_12/lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_12/lstm_28/TensorArrayV2_1z
sequential_12/lstm_28/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/lstm_28/time«
.sequential_12/lstm_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_12/lstm_28/while/maximum_iterations
(sequential_12/lstm_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_12/lstm_28/while/loop_counterÛ
sequential_12/lstm_28/whileWhile1sequential_12/lstm_28/while/loop_counter:output:07sequential_12/lstm_28/while/maximum_iterations:output:0#sequential_12/lstm_28/time:output:0.sequential_12/lstm_28/TensorArrayV2_1:handle:0$sequential_12/lstm_28/zeros:output:0&sequential_12/lstm_28/zeros_1:output:0.sequential_12/lstm_28/strided_slice_1:output:0Msequential_12/lstm_28/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_28_lstm_cell_28_matmul_readvariableop_resourceCsequential_12_lstm_28_lstm_cell_28_matmul_1_readvariableop_resourceBsequential_12_lstm_28_lstm_cell_28_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_12_lstm_28_while_body_1057502*4
cond,R*
(sequential_12_lstm_28_while_cond_1057501*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
sequential_12/lstm_28/whileá
Fsequential_12/lstm_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2H
Fsequential_12/lstm_28/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_12/lstm_28/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_28/while:output:3Osequential_12/lstm_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02:
8sequential_12/lstm_28/TensorArrayV2Stack/TensorListStack­
+sequential_12/lstm_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_12/lstm_28/strided_slice_3/stack¨
-sequential_12/lstm_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_12/lstm_28/strided_slice_3/stack_1¨
-sequential_12/lstm_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_28/strided_slice_3/stack_2
%sequential_12/lstm_28/strided_slice_3StridedSliceAsequential_12/lstm_28/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_28/strided_slice_3/stack:output:06sequential_12/lstm_28/strided_slice_3/stack_1:output:06sequential_12/lstm_28/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2'
%sequential_12/lstm_28/strided_slice_3¥
&sequential_12/lstm_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_28/transpose_1/permþ
!sequential_12/lstm_28/transpose_1	TransposeAsequential_12/lstm_28/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_28/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!sequential_12/lstm_28/transpose_1
sequential_12/lstm_28/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_28/runtime°
!sequential_12/dropout_45/IdentityIdentity%sequential_12/lstm_28/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!sequential_12/dropout_45/IdentityÝ
/sequential_12/dense_29/Tensordot/ReadVariableOpReadVariableOp8sequential_12_dense_29_tensordot_readvariableop_resource* 
_output_shapes
:
ªª*
dtype021
/sequential_12/dense_29/Tensordot/ReadVariableOp
%sequential_12/dense_29/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_12/dense_29/Tensordot/axes
%sequential_12/dense_29/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_12/dense_29/Tensordot/freeª
&sequential_12/dense_29/Tensordot/ShapeShape*sequential_12/dropout_45/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_12/dense_29/Tensordot/Shape¢
.sequential_12/dense_29/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/dense_29/Tensordot/GatherV2/axisÄ
)sequential_12/dense_29/Tensordot/GatherV2GatherV2/sequential_12/dense_29/Tensordot/Shape:output:0.sequential_12/dense_29/Tensordot/free:output:07sequential_12/dense_29/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_12/dense_29/Tensordot/GatherV2¦
0sequential_12/dense_29/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_12/dense_29/Tensordot/GatherV2_1/axisÊ
+sequential_12/dense_29/Tensordot/GatherV2_1GatherV2/sequential_12/dense_29/Tensordot/Shape:output:0.sequential_12/dense_29/Tensordot/axes:output:09sequential_12/dense_29/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_12/dense_29/Tensordot/GatherV2_1
&sequential_12/dense_29/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_12/dense_29/Tensordot/ConstÜ
%sequential_12/dense_29/Tensordot/ProdProd2sequential_12/dense_29/Tensordot/GatherV2:output:0/sequential_12/dense_29/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_12/dense_29/Tensordot/Prod
(sequential_12/dense_29/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_12/dense_29/Tensordot/Const_1ä
'sequential_12/dense_29/Tensordot/Prod_1Prod4sequential_12/dense_29/Tensordot/GatherV2_1:output:01sequential_12/dense_29/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_12/dense_29/Tensordot/Prod_1
,sequential_12/dense_29/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_12/dense_29/Tensordot/concat/axis£
'sequential_12/dense_29/Tensordot/concatConcatV2.sequential_12/dense_29/Tensordot/free:output:0.sequential_12/dense_29/Tensordot/axes:output:05sequential_12/dense_29/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_12/dense_29/Tensordot/concatè
&sequential_12/dense_29/Tensordot/stackPack.sequential_12/dense_29/Tensordot/Prod:output:00sequential_12/dense_29/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_12/dense_29/Tensordot/stackú
*sequential_12/dense_29/Tensordot/transpose	Transpose*sequential_12/dropout_45/Identity:output:00sequential_12/dense_29/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2,
*sequential_12/dense_29/Tensordot/transposeû
(sequential_12/dense_29/Tensordot/ReshapeReshape.sequential_12/dense_29/Tensordot/transpose:y:0/sequential_12/dense_29/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_12/dense_29/Tensordot/Reshapeû
'sequential_12/dense_29/Tensordot/MatMulMatMul1sequential_12/dense_29/Tensordot/Reshape:output:07sequential_12/dense_29/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2)
'sequential_12/dense_29/Tensordot/MatMul
(sequential_12/dense_29/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ª2*
(sequential_12/dense_29/Tensordot/Const_2¢
.sequential_12/dense_29/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/dense_29/Tensordot/concat_1/axis°
)sequential_12/dense_29/Tensordot/concat_1ConcatV22sequential_12/dense_29/Tensordot/GatherV2:output:01sequential_12/dense_29/Tensordot/Const_2:output:07sequential_12/dense_29/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_12/dense_29/Tensordot/concat_1í
 sequential_12/dense_29/TensordotReshape1sequential_12/dense_29/Tensordot/MatMul:product:02sequential_12/dense_29/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 sequential_12/dense_29/TensordotÒ
-sequential_12/dense_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_29_biasadd_readvariableop_resource*
_output_shapes	
:ª*
dtype02/
-sequential_12/dense_29/BiasAdd/ReadVariableOpä
sequential_12/dense_29/BiasAddBiasAdd)sequential_12/dense_29/Tensordot:output:05sequential_12/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
sequential_12/dense_29/BiasAdd¢
sequential_12/dense_29/ReluRelu'sequential_12/dense_29/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
sequential_12/dense_29/Relu´
!sequential_12/dropout_46/IdentityIdentity)sequential_12/dense_29/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!sequential_12/dropout_46/IdentityÜ
/sequential_12/dense_30/Tensordot/ReadVariableOpReadVariableOp8sequential_12_dense_30_tensordot_readvariableop_resource*
_output_shapes
:	ª*
dtype021
/sequential_12/dense_30/Tensordot/ReadVariableOp
%sequential_12/dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_12/dense_30/Tensordot/axes
%sequential_12/dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_12/dense_30/Tensordot/freeª
&sequential_12/dense_30/Tensordot/ShapeShape*sequential_12/dropout_46/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_12/dense_30/Tensordot/Shape¢
.sequential_12/dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/dense_30/Tensordot/GatherV2/axisÄ
)sequential_12/dense_30/Tensordot/GatherV2GatherV2/sequential_12/dense_30/Tensordot/Shape:output:0.sequential_12/dense_30/Tensordot/free:output:07sequential_12/dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_12/dense_30/Tensordot/GatherV2¦
0sequential_12/dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_12/dense_30/Tensordot/GatherV2_1/axisÊ
+sequential_12/dense_30/Tensordot/GatherV2_1GatherV2/sequential_12/dense_30/Tensordot/Shape:output:0.sequential_12/dense_30/Tensordot/axes:output:09sequential_12/dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_12/dense_30/Tensordot/GatherV2_1
&sequential_12/dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_12/dense_30/Tensordot/ConstÜ
%sequential_12/dense_30/Tensordot/ProdProd2sequential_12/dense_30/Tensordot/GatherV2:output:0/sequential_12/dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_12/dense_30/Tensordot/Prod
(sequential_12/dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_12/dense_30/Tensordot/Const_1ä
'sequential_12/dense_30/Tensordot/Prod_1Prod4sequential_12/dense_30/Tensordot/GatherV2_1:output:01sequential_12/dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_12/dense_30/Tensordot/Prod_1
,sequential_12/dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_12/dense_30/Tensordot/concat/axis£
'sequential_12/dense_30/Tensordot/concatConcatV2.sequential_12/dense_30/Tensordot/free:output:0.sequential_12/dense_30/Tensordot/axes:output:05sequential_12/dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_12/dense_30/Tensordot/concatè
&sequential_12/dense_30/Tensordot/stackPack.sequential_12/dense_30/Tensordot/Prod:output:00sequential_12/dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_12/dense_30/Tensordot/stackú
*sequential_12/dense_30/Tensordot/transpose	Transpose*sequential_12/dropout_46/Identity:output:00sequential_12/dense_30/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2,
*sequential_12/dense_30/Tensordot/transposeû
(sequential_12/dense_30/Tensordot/ReshapeReshape.sequential_12/dense_30/Tensordot/transpose:y:0/sequential_12/dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_12/dense_30/Tensordot/Reshapeú
'sequential_12/dense_30/Tensordot/MatMulMatMul1sequential_12/dense_30/Tensordot/Reshape:output:07sequential_12/dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_12/dense_30/Tensordot/MatMul
(sequential_12/dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_12/dense_30/Tensordot/Const_2¢
.sequential_12/dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/dense_30/Tensordot/concat_1/axis°
)sequential_12/dense_30/Tensordot/concat_1ConcatV22sequential_12/dense_30/Tensordot/GatherV2:output:01sequential_12/dense_30/Tensordot/Const_2:output:07sequential_12/dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_12/dense_30/Tensordot/concat_1ì
 sequential_12/dense_30/TensordotReshape1sequential_12/dense_30/Tensordot/MatMul:product:02sequential_12/dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_12/dense_30/TensordotÑ
-sequential_12/dense_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_12/dense_30/BiasAdd/ReadVariableOpã
sequential_12/dense_30/BiasAddBiasAdd)sequential_12/dense_30/Tensordot:output:05sequential_12/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_12/dense_30/BiasAdd
IdentityIdentity'sequential_12/dense_30/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp.^sequential_12/dense_29/BiasAdd/ReadVariableOp0^sequential_12/dense_29/Tensordot/ReadVariableOp.^sequential_12/dense_30/BiasAdd/ReadVariableOp0^sequential_12/dense_30/Tensordot/ReadVariableOp:^sequential_12/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp9^sequential_12/lstm_26/lstm_cell_26/MatMul/ReadVariableOp;^sequential_12/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^sequential_12/lstm_26/while:^sequential_12/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp9^sequential_12/lstm_27/lstm_cell_27/MatMul/ReadVariableOp;^sequential_12/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^sequential_12/lstm_27/while:^sequential_12/lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp9^sequential_12/lstm_28/lstm_cell_28/MatMul/ReadVariableOp;^sequential_12/lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp^sequential_12/lstm_28/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2^
-sequential_12/dense_29/BiasAdd/ReadVariableOp-sequential_12/dense_29/BiasAdd/ReadVariableOp2b
/sequential_12/dense_29/Tensordot/ReadVariableOp/sequential_12/dense_29/Tensordot/ReadVariableOp2^
-sequential_12/dense_30/BiasAdd/ReadVariableOp-sequential_12/dense_30/BiasAdd/ReadVariableOp2b
/sequential_12/dense_30/Tensordot/ReadVariableOp/sequential_12/dense_30/Tensordot/ReadVariableOp2v
9sequential_12/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp9sequential_12/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp2t
8sequential_12/lstm_26/lstm_cell_26/MatMul/ReadVariableOp8sequential_12/lstm_26/lstm_cell_26/MatMul/ReadVariableOp2x
:sequential_12/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:sequential_12/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp2:
sequential_12/lstm_26/whilesequential_12/lstm_26/while2v
9sequential_12/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp9sequential_12/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2t
8sequential_12/lstm_27/lstm_cell_27/MatMul/ReadVariableOp8sequential_12/lstm_27/lstm_cell_27/MatMul/ReadVariableOp2x
:sequential_12/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:sequential_12/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2:
sequential_12/lstm_27/whilesequential_12/lstm_27/while2v
9sequential_12/lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp9sequential_12/lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp2t
8sequential_12/lstm_28/lstm_cell_28/MatMul/ReadVariableOp8sequential_12/lstm_28/lstm_cell_28/MatMul/ReadVariableOp2x
:sequential_12/lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp:sequential_12/lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp2:
sequential_12/lstm_28/whilesequential_12/lstm_28/while:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_26_input
³?
Õ
while_body_1063457
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_28_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_28_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_28_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_28_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_28_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_28/BiasAdd/ReadVariableOp¢(while/lstm_cell_28/MatMul/ReadVariableOp¢*while/lstm_cell_28/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_28/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_28_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_28/MatMul/ReadVariableOp×
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMulÐ
*while/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_28_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_28/MatMul_1/ReadVariableOpÀ
while/lstm_cell_28/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMul_1¸
while/lstm_cell_28/addAddV2#while/lstm_cell_28/MatMul:product:0%while/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/addÈ
)while/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_28_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_28/BiasAdd/ReadVariableOpÅ
while/lstm_cell_28/BiasAddBiasAddwhile/lstm_cell_28/add:z:01while/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/BiasAdd
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_28/split/split_dim
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0#while/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_28/split
while/lstm_cell_28/SigmoidSigmoid!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid
while/lstm_cell_28/Sigmoid_1Sigmoid!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_1¡
while/lstm_cell_28/mulMul while/lstm_cell_28/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul
while/lstm_cell_28/ReluRelu!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Reluµ
while/lstm_cell_28/mul_1Mulwhile/lstm_cell_28/Sigmoid:y:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_1ª
while/lstm_cell_28/add_1AddV2while/lstm_cell_28/mul:z:0while/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/add_1
while/lstm_cell_28/Sigmoid_2Sigmoid!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_2
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Relu_1¹
while/lstm_cell_28/mul_2Mul while/lstm_cell_28/Sigmoid_2:y:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_28/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_28/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_28/BiasAdd/ReadVariableOp)^while/lstm_cell_28/MatMul/ReadVariableOp+^while/lstm_cell_28/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_28_biasadd_readvariableop_resource4while_lstm_cell_28_biasadd_readvariableop_resource_0"l
3while_lstm_cell_28_matmul_1_readvariableop_resource5while_lstm_cell_28_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_28_matmul_readvariableop_resource3while_lstm_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_28/BiasAdd/ReadVariableOp)while/lstm_cell_28/BiasAdd/ReadVariableOp2T
(while/lstm_cell_28/MatMul/ReadVariableOp(while/lstm_cell_28/MatMul/ReadVariableOp2X
*while/lstm_cell_28/MatMul_1/ReadVariableOp*while/lstm_cell_28/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 

e
G__inference_dropout_44_layer_call_and_return_conditional_losses_1063199

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Þ
È
while_cond_1062027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1062027___redundant_placeholder05
1while_while_cond_1062027___redundant_placeholder15
1while_while_cond_1062027___redundant_placeholder25
1while_while_cond_1062027___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
èJ
Õ

lstm_27_while_body_1061625,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3+
'lstm_27_while_lstm_27_strided_slice_1_0g
clstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:
ª¨Q
=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ª¨K
<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	¨
lstm_27_while_identity
lstm_27_while_identity_1
lstm_27_while_identity_2
lstm_27_while_identity_3
lstm_27_while_identity_4
lstm_27_while_identity_5)
%lstm_27_while_lstm_27_strided_slice_1e
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorM
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:
ª¨O
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨I
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	¨¢1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp¢0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp¢2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpÓ
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2A
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0lstm_27_while_placeholderHlstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype023
1lstm_27/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype022
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp÷
!lstm_27/while/lstm_cell_27/MatMulMatMul8lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2#
!lstm_27/while/lstm_cell_27/MatMulè
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype024
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpà
#lstm_27/while/lstm_cell_27/MatMul_1MatMullstm_27_while_placeholder_2:lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2%
#lstm_27/while/lstm_cell_27/MatMul_1Ø
lstm_27/while/lstm_cell_27/addAddV2+lstm_27/while/lstm_cell_27/MatMul:product:0-lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2 
lstm_27/while/lstm_cell_27/addà
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype023
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpå
"lstm_27/while/lstm_cell_27/BiasAddBiasAdd"lstm_27/while/lstm_cell_27/add:z:09lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2$
"lstm_27/while/lstm_cell_27/BiasAdd
*lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_27/while/lstm_cell_27/split/split_dim¯
 lstm_27/while/lstm_cell_27/splitSplit3lstm_27/while/lstm_cell_27/split/split_dim:output:0+lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2"
 lstm_27/while/lstm_cell_27/split±
"lstm_27/while/lstm_cell_27/SigmoidSigmoid)lstm_27/while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2$
"lstm_27/while/lstm_cell_27/Sigmoidµ
$lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid)lstm_27/while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_27/while/lstm_cell_27/Sigmoid_1Á
lstm_27/while/lstm_cell_27/mulMul(lstm_27/while/lstm_cell_27/Sigmoid_1:y:0lstm_27_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_27/while/lstm_cell_27/mul¨
lstm_27/while/lstm_cell_27/ReluRelu)lstm_27/while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
lstm_27/while/lstm_cell_27/ReluÕ
 lstm_27/while/lstm_cell_27/mul_1Mul&lstm_27/while/lstm_cell_27/Sigmoid:y:0-lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_27/while/lstm_cell_27/mul_1Ê
 lstm_27/while/lstm_cell_27/add_1AddV2"lstm_27/while/lstm_cell_27/mul:z:0$lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_27/while/lstm_cell_27/add_1µ
$lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid)lstm_27/while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_27/while/lstm_cell_27/Sigmoid_2§
!lstm_27/while/lstm_cell_27/Relu_1Relu$lstm_27/while/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!lstm_27/while/lstm_cell_27/Relu_1Ù
 lstm_27/while/lstm_cell_27/mul_2Mul(lstm_27/while/lstm_cell_27/Sigmoid_2:y:0/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_27/while/lstm_cell_27/mul_2
2lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_27_while_placeholder_1lstm_27_while_placeholder$lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_27/while/TensorArrayV2Write/TensorListSetIteml
lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_27/while/add/y
lstm_27/while/addAddV2lstm_27_while_placeholderlstm_27/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_27/while/addp
lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_27/while/add_1/y
lstm_27/while/add_1AddV2(lstm_27_while_lstm_27_while_loop_counterlstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_27/while/add_1
lstm_27/while/IdentityIdentitylstm_27/while/add_1:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity¦
lstm_27/while/Identity_1Identity.lstm_27_while_lstm_27_while_maximum_iterations^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_1
lstm_27/while/Identity_2Identitylstm_27/while/add:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_2º
lstm_27/while/Identity_3IdentityBlstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_3®
lstm_27/while/Identity_4Identity$lstm_27/while/lstm_cell_27/mul_2:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/while/Identity_4®
lstm_27/while/Identity_5Identity$lstm_27/while/lstm_cell_27/add_1:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/while/Identity_5
lstm_27/while/NoOpNoOp2^lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1^lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp3^lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_27/while/NoOp"9
lstm_27_while_identitylstm_27/while/Identity:output:0"=
lstm_27_while_identity_1!lstm_27/while/Identity_1:output:0"=
lstm_27_while_identity_2!lstm_27/while/Identity_2:output:0"=
lstm_27_while_identity_3!lstm_27/while/Identity_3:output:0"=
lstm_27_while_identity_4!lstm_27/while/Identity_4:output:0"=
lstm_27_while_identity_5!lstm_27/while/Identity_5:output:0"P
%lstm_27_while_lstm_27_strided_slice_1'lstm_27_while_lstm_27_strided_slice_1_0"z
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"|
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"x
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"È
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2f
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2d
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2h
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
³?
Õ
while_body_1063314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_28_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_28_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_28_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_28_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_28_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_28/BiasAdd/ReadVariableOp¢(while/lstm_cell_28/MatMul/ReadVariableOp¢*while/lstm_cell_28/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_28/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_28_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_28/MatMul/ReadVariableOp×
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMulÐ
*while/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_28_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_28/MatMul_1/ReadVariableOpÀ
while/lstm_cell_28/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMul_1¸
while/lstm_cell_28/addAddV2#while/lstm_cell_28/MatMul:product:0%while/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/addÈ
)while/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_28_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_28/BiasAdd/ReadVariableOpÅ
while/lstm_cell_28/BiasAddBiasAddwhile/lstm_cell_28/add:z:01while/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/BiasAdd
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_28/split/split_dim
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0#while/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_28/split
while/lstm_cell_28/SigmoidSigmoid!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid
while/lstm_cell_28/Sigmoid_1Sigmoid!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_1¡
while/lstm_cell_28/mulMul while/lstm_cell_28/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul
while/lstm_cell_28/ReluRelu!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Reluµ
while/lstm_cell_28/mul_1Mulwhile/lstm_cell_28/Sigmoid:y:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_1ª
while/lstm_cell_28/add_1AddV2while/lstm_cell_28/mul:z:0while/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/add_1
while/lstm_cell_28/Sigmoid_2Sigmoid!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_2
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Relu_1¹
while/lstm_cell_28/mul_2Mul while/lstm_cell_28/Sigmoid_2:y:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_28/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_28/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_28/BiasAdd/ReadVariableOp)^while/lstm_cell_28/MatMul/ReadVariableOp+^while/lstm_cell_28/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_28_biasadd_readvariableop_resource4while_lstm_cell_28_biasadd_readvariableop_resource_0"l
3while_lstm_cell_28_matmul_1_readvariableop_resource5while_lstm_cell_28_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_28_matmul_readvariableop_resource3while_lstm_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_28/BiasAdd/ReadVariableOp)while/lstm_cell_28/BiasAdd/ReadVariableOp2T
(while/lstm_cell_28/MatMul/ReadVariableOp(while/lstm_cell_28/MatMul/ReadVariableOp2X
*while/lstm_cell_28/MatMul_1/ReadVariableOp*while/lstm_cell_28/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Ï

è
lstm_27_while_cond_1061624,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3.
*lstm_27_while_less_lstm_27_strided_slice_1E
Alstm_27_while_lstm_27_while_cond_1061624___redundant_placeholder0E
Alstm_27_while_lstm_27_while_cond_1061624___redundant_placeholder1E
Alstm_27_while_lstm_27_while_cond_1061624___redundant_placeholder2E
Alstm_27_while_lstm_27_while_cond_1061624___redundant_placeholder3
lstm_27_while_identity

lstm_27/while/LessLesslstm_27_while_placeholder*lstm_27_while_less_lstm_27_strided_slice_1*
T0*
_output_shapes
: 2
lstm_27/while/Lessu
lstm_27/while/IdentityIdentitylstm_27/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_27/while/Identity"9
lstm_27_while_identitylstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
Ï

è
lstm_26_while_cond_1061477,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3.
*lstm_26_while_less_lstm_26_strided_slice_1E
Alstm_26_while_lstm_26_while_cond_1061477___redundant_placeholder0E
Alstm_26_while_lstm_26_while_cond_1061477___redundant_placeholder1E
Alstm_26_while_lstm_26_while_cond_1061477___redundant_placeholder2E
Alstm_26_while_lstm_26_while_cond_1061477___redundant_placeholder3
lstm_26_while_identity

lstm_26/while/LessLesslstm_26_while_placeholder*lstm_26_while_less_lstm_26_strided_slice_1*
T0*
_output_shapes
: 2
lstm_26/while/Lessu
lstm_26/while/IdentityIdentitylstm_26/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_26/while/Identity"9
lstm_26_while_identitylstm_26/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
Üî
½
J__inference_sequential_12_layer_call_and_return_conditional_losses_1061419

inputsF
3lstm_26_lstm_cell_26_matmul_readvariableop_resource:	¨I
5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨C
4lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	¨G
3lstm_27_lstm_cell_27_matmul_readvariableop_resource:
ª¨I
5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨C
4lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	¨G
3lstm_28_lstm_cell_28_matmul_readvariableop_resource:
ª¨I
5lstm_28_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨C
4lstm_28_lstm_cell_28_biasadd_readvariableop_resource:	¨>
*dense_29_tensordot_readvariableop_resource:
ªª7
(dense_29_biasadd_readvariableop_resource:	ª=
*dense_30_tensordot_readvariableop_resource:	ª6
(dense_30_biasadd_readvariableop_resource:
identity¢dense_29/BiasAdd/ReadVariableOp¢!dense_29/Tensordot/ReadVariableOp¢dense_30/BiasAdd/ReadVariableOp¢!dense_30/Tensordot/ReadVariableOp¢+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp¢*lstm_26/lstm_cell_26/MatMul/ReadVariableOp¢,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp¢lstm_26/while¢+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp¢*lstm_27/lstm_cell_27/MatMul/ReadVariableOp¢,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp¢lstm_27/while¢+lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp¢*lstm_28/lstm_cell_28/MatMul/ReadVariableOp¢,lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp¢lstm_28/whileT
lstm_26/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_26/Shape
lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice/stack
lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_1
lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_2
lstm_26/strided_sliceStridedSlicelstm_26/Shape:output:0$lstm_26/strided_slice/stack:output:0&lstm_26/strided_slice/stack_1:output:0&lstm_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_26/strided_slices
lstm_26/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_26/zeros/packed/1£
lstm_26/zeros/packedPacklstm_26/strided_slice:output:0lstm_26/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_26/zeros/packedo
lstm_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/zeros/Const
lstm_26/zerosFilllstm_26/zeros/packed:output:0lstm_26/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/zerosw
lstm_26/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_26/zeros_1/packed/1©
lstm_26/zeros_1/packedPacklstm_26/strided_slice:output:0!lstm_26/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_26/zeros_1/packeds
lstm_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/zeros_1/Const
lstm_26/zeros_1Filllstm_26/zeros_1/packed:output:0lstm_26/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/zeros_1
lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose/perm
lstm_26/transpose	Transposeinputslstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_26/transposeg
lstm_26/Shape_1Shapelstm_26/transpose:y:0*
T0*
_output_shapes
:2
lstm_26/Shape_1
lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_1/stack
lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_1
lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_2
lstm_26/strided_slice_1StridedSlicelstm_26/Shape_1:output:0&lstm_26/strided_slice_1/stack:output:0(lstm_26/strided_slice_1/stack_1:output:0(lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_26/strided_slice_1
#lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_26/TensorArrayV2/element_shapeÒ
lstm_26/TensorArrayV2TensorListReserve,lstm_26/TensorArrayV2/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_26/TensorArrayV2Ï
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_26/transpose:y:0Flstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_26/TensorArrayUnstack/TensorListFromTensor
lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_2/stack
lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_1
lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_2¬
lstm_26/strided_slice_2StridedSlicelstm_26/transpose:y:0&lstm_26/strided_slice_2/stack:output:0(lstm_26/strided_slice_2/stack_1:output:0(lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_26/strided_slice_2Í
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02,
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpÍ
lstm_26/lstm_cell_26/MatMulMatMul lstm_26/strided_slice_2:output:02lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_26/lstm_cell_26/MatMulÔ
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02.
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpÉ
lstm_26/lstm_cell_26/MatMul_1MatMullstm_26/zeros:output:04lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_26/lstm_cell_26/MatMul_1À
lstm_26/lstm_cell_26/addAddV2%lstm_26/lstm_cell_26/MatMul:product:0'lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_26/lstm_cell_26/addÌ
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02-
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpÍ
lstm_26/lstm_cell_26/BiasAddBiasAddlstm_26/lstm_cell_26/add:z:03lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_26/lstm_cell_26/BiasAdd
$lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_26/lstm_cell_26/split/split_dim
lstm_26/lstm_cell_26/splitSplit-lstm_26/lstm_cell_26/split/split_dim:output:0%lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_26/lstm_cell_26/split
lstm_26/lstm_cell_26/SigmoidSigmoid#lstm_26/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/Sigmoid£
lstm_26/lstm_cell_26/Sigmoid_1Sigmoid#lstm_26/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_26/lstm_cell_26/Sigmoid_1¬
lstm_26/lstm_cell_26/mulMul"lstm_26/lstm_cell_26/Sigmoid_1:y:0lstm_26/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/mul
lstm_26/lstm_cell_26/ReluRelu#lstm_26/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/Relu½
lstm_26/lstm_cell_26/mul_1Mul lstm_26/lstm_cell_26/Sigmoid:y:0'lstm_26/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/mul_1²
lstm_26/lstm_cell_26/add_1AddV2lstm_26/lstm_cell_26/mul:z:0lstm_26/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/add_1£
lstm_26/lstm_cell_26/Sigmoid_2Sigmoid#lstm_26/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_26/lstm_cell_26/Sigmoid_2
lstm_26/lstm_cell_26/Relu_1Relulstm_26/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/Relu_1Á
lstm_26/lstm_cell_26/mul_2Mul"lstm_26/lstm_cell_26/Sigmoid_2:y:0)lstm_26/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/mul_2
%lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2'
%lstm_26/TensorArrayV2_1/element_shapeØ
lstm_26/TensorArrayV2_1TensorListReserve.lstm_26/TensorArrayV2_1/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_26/TensorArrayV2_1^
lstm_26/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_26/time
 lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_26/while/maximum_iterationsz
lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_26/while/loop_counter
lstm_26/whileWhile#lstm_26/while/loop_counter:output:0)lstm_26/while/maximum_iterations:output:0lstm_26/time:output:0 lstm_26/TensorArrayV2_1:handle:0lstm_26/zeros:output:0lstm_26/zeros_1:output:0 lstm_26/strided_slice_1:output:0?lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_26_lstm_cell_26_matmul_readvariableop_resource5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_26_while_body_1061000*&
condR
lstm_26_while_cond_1060999*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
lstm_26/whileÅ
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2:
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_26/TensorArrayV2Stack/TensorListStackTensorListStacklstm_26/while:output:3Alstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02,
*lstm_26/TensorArrayV2Stack/TensorListStack
lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_26/strided_slice_3/stack
lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_26/strided_slice_3/stack_1
lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_3/stack_2Ë
lstm_26/strided_slice_3StridedSlice3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_26/strided_slice_3/stack:output:0(lstm_26/strided_slice_3/stack_1:output:0(lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
lstm_26/strided_slice_3
lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose_1/permÆ
lstm_26/transpose_1	Transpose3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_26/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/transpose_1v
lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/runtime
dropout_43/IdentityIdentitylstm_26/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_43/Identityj
lstm_27/ShapeShapedropout_43/Identity:output:0*
T0*
_output_shapes
:2
lstm_27/Shape
lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice/stack
lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_1
lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_2
lstm_27/strided_sliceStridedSlicelstm_27/Shape:output:0$lstm_27/strided_slice/stack:output:0&lstm_27/strided_slice/stack_1:output:0&lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_27/strided_slices
lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_27/zeros/packed/1£
lstm_27/zeros/packedPacklstm_27/strided_slice:output:0lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_27/zeros/packedo
lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/zeros/Const
lstm_27/zerosFilllstm_27/zeros/packed:output:0lstm_27/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/zerosw
lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_27/zeros_1/packed/1©
lstm_27/zeros_1/packedPacklstm_27/strided_slice:output:0!lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_27/zeros_1/packeds
lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/zeros_1/Const
lstm_27/zeros_1Filllstm_27/zeros_1/packed:output:0lstm_27/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/zeros_1
lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose/perm©
lstm_27/transpose	Transposedropout_43/Identity:output:0lstm_27/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/transposeg
lstm_27/Shape_1Shapelstm_27/transpose:y:0*
T0*
_output_shapes
:2
lstm_27/Shape_1
lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_1/stack
lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_1
lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_2
lstm_27/strided_slice_1StridedSlicelstm_27/Shape_1:output:0&lstm_27/strided_slice_1/stack:output:0(lstm_27/strided_slice_1/stack_1:output:0(lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_27/strided_slice_1
#lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_27/TensorArrayV2/element_shapeÒ
lstm_27/TensorArrayV2TensorListReserve,lstm_27/TensorArrayV2/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_27/TensorArrayV2Ï
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2?
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_27/transpose:y:0Flstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_27/TensorArrayUnstack/TensorListFromTensor
lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_2/stack
lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_1
lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_2­
lstm_27/strided_slice_2StridedSlicelstm_27/transpose:y:0&lstm_27/strided_slice_2/stack:output:0(lstm_27/strided_slice_2/stack_1:output:0(lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
lstm_27/strided_slice_2Î
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3lstm_27_lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02,
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpÍ
lstm_27/lstm_cell_27/MatMulMatMul lstm_27/strided_slice_2:output:02lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_27/lstm_cell_27/MatMulÔ
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02.
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpÉ
lstm_27/lstm_cell_27/MatMul_1MatMullstm_27/zeros:output:04lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_27/lstm_cell_27/MatMul_1À
lstm_27/lstm_cell_27/addAddV2%lstm_27/lstm_cell_27/MatMul:product:0'lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_27/lstm_cell_27/addÌ
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02-
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpÍ
lstm_27/lstm_cell_27/BiasAddBiasAddlstm_27/lstm_cell_27/add:z:03lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_27/lstm_cell_27/BiasAdd
$lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_27/lstm_cell_27/split/split_dim
lstm_27/lstm_cell_27/splitSplit-lstm_27/lstm_cell_27/split/split_dim:output:0%lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_27/lstm_cell_27/split
lstm_27/lstm_cell_27/SigmoidSigmoid#lstm_27/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/Sigmoid£
lstm_27/lstm_cell_27/Sigmoid_1Sigmoid#lstm_27/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_27/lstm_cell_27/Sigmoid_1¬
lstm_27/lstm_cell_27/mulMul"lstm_27/lstm_cell_27/Sigmoid_1:y:0lstm_27/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/mul
lstm_27/lstm_cell_27/ReluRelu#lstm_27/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/Relu½
lstm_27/lstm_cell_27/mul_1Mul lstm_27/lstm_cell_27/Sigmoid:y:0'lstm_27/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/mul_1²
lstm_27/lstm_cell_27/add_1AddV2lstm_27/lstm_cell_27/mul:z:0lstm_27/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/add_1£
lstm_27/lstm_cell_27/Sigmoid_2Sigmoid#lstm_27/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_27/lstm_cell_27/Sigmoid_2
lstm_27/lstm_cell_27/Relu_1Relulstm_27/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/Relu_1Á
lstm_27/lstm_cell_27/mul_2Mul"lstm_27/lstm_cell_27/Sigmoid_2:y:0)lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/mul_2
%lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2'
%lstm_27/TensorArrayV2_1/element_shapeØ
lstm_27/TensorArrayV2_1TensorListReserve.lstm_27/TensorArrayV2_1/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_27/TensorArrayV2_1^
lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_27/time
 lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_27/while/maximum_iterationsz
lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_27/while/loop_counter
lstm_27/whileWhile#lstm_27/while/loop_counter:output:0)lstm_27/while/maximum_iterations:output:0lstm_27/time:output:0 lstm_27/TensorArrayV2_1:handle:0lstm_27/zeros:output:0lstm_27/zeros_1:output:0 lstm_27/strided_slice_1:output:0?lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_27_lstm_cell_27_matmul_readvariableop_resource5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_27_while_body_1061140*&
condR
lstm_27_while_cond_1061139*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
lstm_27/whileÅ
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2:
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_27/TensorArrayV2Stack/TensorListStackTensorListStacklstm_27/while:output:3Alstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02,
*lstm_27/TensorArrayV2Stack/TensorListStack
lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_27/strided_slice_3/stack
lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_27/strided_slice_3/stack_1
lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_3/stack_2Ë
lstm_27/strided_slice_3StridedSlice3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_27/strided_slice_3/stack:output:0(lstm_27/strided_slice_3/stack_1:output:0(lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
lstm_27/strided_slice_3
lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose_1/permÆ
lstm_27/transpose_1	Transpose3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_27/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/transpose_1v
lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/runtime
dropout_44/IdentityIdentitylstm_27/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_44/Identityj
lstm_28/ShapeShapedropout_44/Identity:output:0*
T0*
_output_shapes
:2
lstm_28/Shape
lstm_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_28/strided_slice/stack
lstm_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_28/strided_slice/stack_1
lstm_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_28/strided_slice/stack_2
lstm_28/strided_sliceStridedSlicelstm_28/Shape:output:0$lstm_28/strided_slice/stack:output:0&lstm_28/strided_slice/stack_1:output:0&lstm_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_28/strided_slices
lstm_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_28/zeros/packed/1£
lstm_28/zeros/packedPacklstm_28/strided_slice:output:0lstm_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_28/zeros/packedo
lstm_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_28/zeros/Const
lstm_28/zerosFilllstm_28/zeros/packed:output:0lstm_28/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/zerosw
lstm_28/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_28/zeros_1/packed/1©
lstm_28/zeros_1/packedPacklstm_28/strided_slice:output:0!lstm_28/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_28/zeros_1/packeds
lstm_28/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_28/zeros_1/Const
lstm_28/zeros_1Filllstm_28/zeros_1/packed:output:0lstm_28/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/zeros_1
lstm_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_28/transpose/perm©
lstm_28/transpose	Transposedropout_44/Identity:output:0lstm_28/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/transposeg
lstm_28/Shape_1Shapelstm_28/transpose:y:0*
T0*
_output_shapes
:2
lstm_28/Shape_1
lstm_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_28/strided_slice_1/stack
lstm_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_28/strided_slice_1/stack_1
lstm_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_28/strided_slice_1/stack_2
lstm_28/strided_slice_1StridedSlicelstm_28/Shape_1:output:0&lstm_28/strided_slice_1/stack:output:0(lstm_28/strided_slice_1/stack_1:output:0(lstm_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_28/strided_slice_1
#lstm_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_28/TensorArrayV2/element_shapeÒ
lstm_28/TensorArrayV2TensorListReserve,lstm_28/TensorArrayV2/element_shape:output:0 lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_28/TensorArrayV2Ï
=lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2?
=lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_28/transpose:y:0Flstm_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_28/TensorArrayUnstack/TensorListFromTensor
lstm_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_28/strided_slice_2/stack
lstm_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_28/strided_slice_2/stack_1
lstm_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_28/strided_slice_2/stack_2­
lstm_28/strided_slice_2StridedSlicelstm_28/transpose:y:0&lstm_28/strided_slice_2/stack:output:0(lstm_28/strided_slice_2/stack_1:output:0(lstm_28/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
lstm_28/strided_slice_2Î
*lstm_28/lstm_cell_28/MatMul/ReadVariableOpReadVariableOp3lstm_28_lstm_cell_28_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02,
*lstm_28/lstm_cell_28/MatMul/ReadVariableOpÍ
lstm_28/lstm_cell_28/MatMulMatMul lstm_28/strided_slice_2:output:02lstm_28/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_28/lstm_cell_28/MatMulÔ
,lstm_28/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp5lstm_28_lstm_cell_28_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02.
,lstm_28/lstm_cell_28/MatMul_1/ReadVariableOpÉ
lstm_28/lstm_cell_28/MatMul_1MatMullstm_28/zeros:output:04lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_28/lstm_cell_28/MatMul_1À
lstm_28/lstm_cell_28/addAddV2%lstm_28/lstm_cell_28/MatMul:product:0'lstm_28/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_28/lstm_cell_28/addÌ
+lstm_28/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp4lstm_28_lstm_cell_28_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02-
+lstm_28/lstm_cell_28/BiasAdd/ReadVariableOpÍ
lstm_28/lstm_cell_28/BiasAddBiasAddlstm_28/lstm_cell_28/add:z:03lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_28/lstm_cell_28/BiasAdd
$lstm_28/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_28/lstm_cell_28/split/split_dim
lstm_28/lstm_cell_28/splitSplit-lstm_28/lstm_cell_28/split/split_dim:output:0%lstm_28/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_28/lstm_cell_28/split
lstm_28/lstm_cell_28/SigmoidSigmoid#lstm_28/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/Sigmoid£
lstm_28/lstm_cell_28/Sigmoid_1Sigmoid#lstm_28/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_28/lstm_cell_28/Sigmoid_1¬
lstm_28/lstm_cell_28/mulMul"lstm_28/lstm_cell_28/Sigmoid_1:y:0lstm_28/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/mul
lstm_28/lstm_cell_28/ReluRelu#lstm_28/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/Relu½
lstm_28/lstm_cell_28/mul_1Mul lstm_28/lstm_cell_28/Sigmoid:y:0'lstm_28/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/mul_1²
lstm_28/lstm_cell_28/add_1AddV2lstm_28/lstm_cell_28/mul:z:0lstm_28/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/add_1£
lstm_28/lstm_cell_28/Sigmoid_2Sigmoid#lstm_28/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_28/lstm_cell_28/Sigmoid_2
lstm_28/lstm_cell_28/Relu_1Relulstm_28/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/Relu_1Á
lstm_28/lstm_cell_28/mul_2Mul"lstm_28/lstm_cell_28/Sigmoid_2:y:0)lstm_28/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/mul_2
%lstm_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2'
%lstm_28/TensorArrayV2_1/element_shapeØ
lstm_28/TensorArrayV2_1TensorListReserve.lstm_28/TensorArrayV2_1/element_shape:output:0 lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_28/TensorArrayV2_1^
lstm_28/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_28/time
 lstm_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_28/while/maximum_iterationsz
lstm_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_28/while/loop_counter
lstm_28/whileWhile#lstm_28/while/loop_counter:output:0)lstm_28/while/maximum_iterations:output:0lstm_28/time:output:0 lstm_28/TensorArrayV2_1:handle:0lstm_28/zeros:output:0lstm_28/zeros_1:output:0 lstm_28/strided_slice_1:output:0?lstm_28/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_28_lstm_cell_28_matmul_readvariableop_resource5lstm_28_lstm_cell_28_matmul_1_readvariableop_resource4lstm_28_lstm_cell_28_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_28_while_body_1061280*&
condR
lstm_28_while_cond_1061279*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
lstm_28/whileÅ
8lstm_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2:
8lstm_28/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_28/TensorArrayV2Stack/TensorListStackTensorListStacklstm_28/while:output:3Alstm_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02,
*lstm_28/TensorArrayV2Stack/TensorListStack
lstm_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_28/strided_slice_3/stack
lstm_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_28/strided_slice_3/stack_1
lstm_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_28/strided_slice_3/stack_2Ë
lstm_28/strided_slice_3StridedSlice3lstm_28/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_28/strided_slice_3/stack:output:0(lstm_28/strided_slice_3/stack_1:output:0(lstm_28/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
lstm_28/strided_slice_3
lstm_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_28/transpose_1/permÆ
lstm_28/transpose_1	Transpose3lstm_28/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_28/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/transpose_1v
lstm_28/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_28/runtime
dropout_45/IdentityIdentitylstm_28/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_45/Identity³
!dense_29/Tensordot/ReadVariableOpReadVariableOp*dense_29_tensordot_readvariableop_resource* 
_output_shapes
:
ªª*
dtype02#
!dense_29/Tensordot/ReadVariableOp|
dense_29/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_29/Tensordot/axes
dense_29/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_29/Tensordot/free
dense_29/Tensordot/ShapeShapedropout_45/Identity:output:0*
T0*
_output_shapes
:2
dense_29/Tensordot/Shape
 dense_29/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/GatherV2/axisþ
dense_29/Tensordot/GatherV2GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/free:output:0)dense_29/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2
"dense_29/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_29/Tensordot/GatherV2_1/axis
dense_29/Tensordot/GatherV2_1GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/axes:output:0+dense_29/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2_1~
dense_29/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const¤
dense_29/Tensordot/ProdProd$dense_29/Tensordot/GatherV2:output:0!dense_29/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod
dense_29/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const_1¬
dense_29/Tensordot/Prod_1Prod&dense_29/Tensordot/GatherV2_1:output:0#dense_29/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod_1
dense_29/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_29/Tensordot/concat/axisÝ
dense_29/Tensordot/concatConcatV2 dense_29/Tensordot/free:output:0 dense_29/Tensordot/axes:output:0'dense_29/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat°
dense_29/Tensordot/stackPack dense_29/Tensordot/Prod:output:0"dense_29/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/stackÂ
dense_29/Tensordot/transpose	Transposedropout_45/Identity:output:0"dense_29/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_29/Tensordot/transposeÃ
dense_29/Tensordot/ReshapeReshape dense_29/Tensordot/transpose:y:0!dense_29/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_29/Tensordot/ReshapeÃ
dense_29/Tensordot/MatMulMatMul#dense_29/Tensordot/Reshape:output:0)dense_29/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_29/Tensordot/MatMul
dense_29/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ª2
dense_29/Tensordot/Const_2
 dense_29/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/concat_1/axisê
dense_29/Tensordot/concat_1ConcatV2$dense_29/Tensordot/GatherV2:output:0#dense_29/Tensordot/Const_2:output:0)dense_29/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat_1µ
dense_29/TensordotReshape#dense_29/Tensordot/MatMul:product:0$dense_29/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_29/Tensordot¨
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes	
:ª*
dtype02!
dense_29/BiasAdd/ReadVariableOp¬
dense_29/BiasAddBiasAdddense_29/Tensordot:output:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_29/BiasAddx
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_29/Relu
dropout_46/IdentityIdentitydense_29/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_46/Identity²
!dense_30/Tensordot/ReadVariableOpReadVariableOp*dense_30_tensordot_readvariableop_resource*
_output_shapes
:	ª*
dtype02#
!dense_30/Tensordot/ReadVariableOp|
dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_30/Tensordot/axes
dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_30/Tensordot/free
dense_30/Tensordot/ShapeShapedropout_46/Identity:output:0*
T0*
_output_shapes
:2
dense_30/Tensordot/Shape
 dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/GatherV2/axisþ
dense_30/Tensordot/GatherV2GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/free:output:0)dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2
"dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_30/Tensordot/GatherV2_1/axis
dense_30/Tensordot/GatherV2_1GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/axes:output:0+dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2_1~
dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const¤
dense_30/Tensordot/ProdProd$dense_30/Tensordot/GatherV2:output:0!dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod
dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const_1¬
dense_30/Tensordot/Prod_1Prod&dense_30/Tensordot/GatherV2_1:output:0#dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod_1
dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_30/Tensordot/concat/axisÝ
dense_30/Tensordot/concatConcatV2 dense_30/Tensordot/free:output:0 dense_30/Tensordot/axes:output:0'dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat°
dense_30/Tensordot/stackPack dense_30/Tensordot/Prod:output:0"dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/stackÂ
dense_30/Tensordot/transpose	Transposedropout_46/Identity:output:0"dense_30/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_30/Tensordot/transposeÃ
dense_30/Tensordot/ReshapeReshape dense_30/Tensordot/transpose:y:0!dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_30/Tensordot/ReshapeÂ
dense_30/Tensordot/MatMulMatMul#dense_30/Tensordot/Reshape:output:0)dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_30/Tensordot/MatMul
dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_30/Tensordot/Const_2
 dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/concat_1/axisê
dense_30/Tensordot/concat_1ConcatV2$dense_30/Tensordot/GatherV2:output:0#dense_30/Tensordot/Const_2:output:0)dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat_1´
dense_30/TensordotReshape#dense_30/Tensordot/MatMul:product:0$dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_30/Tensordot§
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_30/BiasAdd/ReadVariableOp«
dense_30/BiasAddBiasAdddense_30/Tensordot:output:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_30/BiasAddx
IdentityIdentitydense_30/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¨
NoOpNoOp ^dense_29/BiasAdd/ReadVariableOp"^dense_29/Tensordot/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp"^dense_30/Tensordot/ReadVariableOp,^lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+^lstm_26/lstm_cell_26/MatMul/ReadVariableOp-^lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^lstm_26/while,^lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+^lstm_27/lstm_cell_27/MatMul/ReadVariableOp-^lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^lstm_27/while,^lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp+^lstm_28/lstm_cell_28/MatMul/ReadVariableOp-^lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp^lstm_28/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2F
!dense_29/Tensordot/ReadVariableOp!dense_29/Tensordot/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2F
!dense_30/Tensordot/ReadVariableOp!dense_30/Tensordot/ReadVariableOp2Z
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp2X
*lstm_26/lstm_cell_26/MatMul/ReadVariableOp*lstm_26/lstm_cell_26/MatMul/ReadVariableOp2\
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp2
lstm_26/whilelstm_26/while2Z
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2X
*lstm_27/lstm_cell_27/MatMul/ReadVariableOp*lstm_27/lstm_cell_27/MatMul/ReadVariableOp2\
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2
lstm_27/whilelstm_27/while2Z
+lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp+lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp2X
*lstm_28/lstm_cell_28/MatMul/ReadVariableOp*lstm_28/lstm_cell_28/MatMul/ReadVariableOp2\
,lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp,lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp2
lstm_28/whilelstm_28/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
º
)__inference_lstm_27_layer_call_fn_1062579
inputs_0
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_27_layer_call_and_return_conditional_losses_10583892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
inputs/0
·
¸
)__inference_lstm_27_layer_call_fn_1062601

inputs
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_27_layer_call_and_return_conditional_losses_10597412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
ô

(sequential_12_lstm_26_while_cond_1057221H
Dsequential_12_lstm_26_while_sequential_12_lstm_26_while_loop_counterN
Jsequential_12_lstm_26_while_sequential_12_lstm_26_while_maximum_iterations+
'sequential_12_lstm_26_while_placeholder-
)sequential_12_lstm_26_while_placeholder_1-
)sequential_12_lstm_26_while_placeholder_2-
)sequential_12_lstm_26_while_placeholder_3J
Fsequential_12_lstm_26_while_less_sequential_12_lstm_26_strided_slice_1a
]sequential_12_lstm_26_while_sequential_12_lstm_26_while_cond_1057221___redundant_placeholder0a
]sequential_12_lstm_26_while_sequential_12_lstm_26_while_cond_1057221___redundant_placeholder1a
]sequential_12_lstm_26_while_sequential_12_lstm_26_while_cond_1057221___redundant_placeholder2a
]sequential_12_lstm_26_while_sequential_12_lstm_26_while_cond_1057221___redundant_placeholder3(
$sequential_12_lstm_26_while_identity
Þ
 sequential_12/lstm_26/while/LessLess'sequential_12_lstm_26_while_placeholderFsequential_12_lstm_26_while_less_sequential_12_lstm_26_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_12/lstm_26/while/Less
$sequential_12/lstm_26/while/IdentityIdentity$sequential_12/lstm_26/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_12/lstm_26/while/Identity"U
$sequential_12_lstm_26_while_identity-sequential_12/lstm_26/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
á
º
)__inference_lstm_28_layer_call_fn_1063233
inputs_0
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_28_layer_call_and_return_conditional_losses_10591892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
inputs/0


I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_1064124

inputs
states_0
states_12
matmul_readvariableop_resource:
ª¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1
ÿ?

D__inference_lstm_28_layer_call_and_return_conditional_losses_1058987

inputs(
lstm_cell_28_1058905:
ª¨(
lstm_cell_28_1058907:
ª¨#
lstm_cell_28_1058909:	¨
identity¢$lstm_cell_28/StatefulPartitionedCall¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¤
$lstm_cell_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_28_1058905lstm_cell_28_1058907lstm_cell_28_1058909*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_10589042&
$lstm_cell_28/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_28_1058905lstm_cell_28_1058907lstm_cell_28_1058909*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1058918*
condR
while_cond_1058917*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

Identity}
NoOpNoOp%^lstm_cell_28/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 2L
$lstm_cell_28/StatefulPartitionedCall$lstm_cell_28/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs


I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_1064058

inputs
states_0
states_11
matmul_readvariableop_resource:	¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1
Ö!
ÿ
E__inference_dense_29_layer_call_and_return_conditional_losses_1059944

inputs5
!tensordot_readvariableop_resource:
ªª.
biasadd_readvariableop_resource:	ª
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
ªª*
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
:ÿÿÿÿÿÿÿÿÿª2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ª2
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
:ÿÿÿÿÿÿÿÿÿª2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ª*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿª: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Þ
È
while_cond_1062813
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1062813___redundant_placeholder05
1while_while_cond_1062813___redundant_placeholder15
1while_while_cond_1062813___redundant_placeholder25
1while_while_cond_1062813___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:


I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_1064156

inputs
states_0
states_12
matmul_readvariableop_resource:
ª¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1


I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_1059050

inputs

states
states_12
matmul_readvariableop_resource:
ª¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates

e
G__inference_dropout_44_layer_call_and_return_conditional_losses_1059754

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
ÃU

D__inference_lstm_28_layer_call_and_return_conditional_losses_1063684

inputs?
+lstm_cell_28_matmul_readvariableop_resource:
ª¨A
-lstm_cell_28_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_28_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_28/BiasAdd/ReadVariableOp¢"lstm_cell_28/MatMul/ReadVariableOp¢$lstm_cell_28/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
:ÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_28/MatMul/ReadVariableOpReadVariableOp+lstm_cell_28_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_28/MatMul/ReadVariableOp­
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0*lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul¼
$lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_28_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_28/MatMul_1/ReadVariableOp©
lstm_cell_28/MatMul_1MatMulzeros:output:0,lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul_1 
lstm_cell_28/addAddV2lstm_cell_28/MatMul:product:0lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/add´
#lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_28_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_28/BiasAdd/ReadVariableOp­
lstm_cell_28/BiasAddBiasAddlstm_cell_28/add:z:0+lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/BiasAdd~
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_28/split/split_dim÷
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_28/split
lstm_cell_28/SigmoidSigmoidlstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid
lstm_cell_28/Sigmoid_1Sigmoidlstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_1
lstm_cell_28/mulMullstm_cell_28/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul~
lstm_cell_28/ReluRelulstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu
lstm_cell_28/mul_1Mullstm_cell_28/Sigmoid:y:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_1
lstm_cell_28/add_1AddV2lstm_cell_28/mul:z:0lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/add_1
lstm_cell_28/Sigmoid_2Sigmoidlstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_2}
lstm_cell_28/Relu_1Relulstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu_1¡
lstm_cell_28/mul_2Mullstm_cell_28/Sigmoid_2:y:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_28_matmul_readvariableop_resource-lstm_cell_28_matmul_1_readvariableop_resource,lstm_cell_28_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1063600*
condR
while_cond_1063599*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_28/BiasAdd/ReadVariableOp#^lstm_cell_28/MatMul/ReadVariableOp%^lstm_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_28/BiasAdd/ReadVariableOp#lstm_cell_28/BiasAdd/ReadVariableOp2H
"lstm_cell_28/MatMul/ReadVariableOp"lstm_cell_28/MatMul/ReadVariableOp2L
$lstm_cell_28/MatMul_1/ReadVariableOp$lstm_cell_28/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Þ
È
while_cond_1060160
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1060160___redundant_placeholder05
1while_while_cond_1060160___redundant_placeholder15
1while_while_cond_1060160___redundant_placeholder25
1while_while_cond_1060160___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:


I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_1057854

inputs

states
states_11
matmul_readvariableop_resource:	¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates
V
 
D__inference_lstm_27_layer_call_and_return_conditional_losses_1062898
inputs_0?
+lstm_cell_27_matmul_readvariableop_resource:
ª¨A
-lstm_cell_27_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_27_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileF
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1062814*
condR
while_cond_1062813*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
inputs/0
Þ
È
while_cond_1059813
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1059813___redundant_placeholder05
1while_while_cond_1059813___redundant_placeholder15
1while_while_cond_1059813___redundant_placeholder25
1while_while_cond_1059813___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
èJ
Õ

lstm_28_while_body_1061772,
(lstm_28_while_lstm_28_while_loop_counter2
.lstm_28_while_lstm_28_while_maximum_iterations
lstm_28_while_placeholder
lstm_28_while_placeholder_1
lstm_28_while_placeholder_2
lstm_28_while_placeholder_3+
'lstm_28_while_lstm_28_strided_slice_1_0g
clstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_28_while_lstm_cell_28_matmul_readvariableop_resource_0:
ª¨Q
=lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource_0:
ª¨K
<lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource_0:	¨
lstm_28_while_identity
lstm_28_while_identity_1
lstm_28_while_identity_2
lstm_28_while_identity_3
lstm_28_while_identity_4
lstm_28_while_identity_5)
%lstm_28_while_lstm_28_strided_slice_1e
alstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensorM
9lstm_28_while_lstm_cell_28_matmul_readvariableop_resource:
ª¨O
;lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨I
:lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource:	¨¢1lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp¢0lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp¢2lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOpÓ
?lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2A
?lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_28/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0lstm_28_while_placeholderHlstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype023
1lstm_28/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_28/while/lstm_cell_28/MatMul/ReadVariableOpReadVariableOp;lstm_28_while_lstm_cell_28_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype022
0lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp÷
!lstm_28/while/lstm_cell_28/MatMulMatMul8lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2#
!lstm_28/while/lstm_cell_28/MatMulè
2lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp=lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype024
2lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOpà
#lstm_28/while/lstm_cell_28/MatMul_1MatMullstm_28_while_placeholder_2:lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2%
#lstm_28/while/lstm_cell_28/MatMul_1Ø
lstm_28/while/lstm_cell_28/addAddV2+lstm_28/while/lstm_cell_28/MatMul:product:0-lstm_28/while/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2 
lstm_28/while/lstm_cell_28/addà
1lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp<lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype023
1lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOpå
"lstm_28/while/lstm_cell_28/BiasAddBiasAdd"lstm_28/while/lstm_cell_28/add:z:09lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2$
"lstm_28/while/lstm_cell_28/BiasAdd
*lstm_28/while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_28/while/lstm_cell_28/split/split_dim¯
 lstm_28/while/lstm_cell_28/splitSplit3lstm_28/while/lstm_cell_28/split/split_dim:output:0+lstm_28/while/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2"
 lstm_28/while/lstm_cell_28/split±
"lstm_28/while/lstm_cell_28/SigmoidSigmoid)lstm_28/while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2$
"lstm_28/while/lstm_cell_28/Sigmoidµ
$lstm_28/while/lstm_cell_28/Sigmoid_1Sigmoid)lstm_28/while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_28/while/lstm_cell_28/Sigmoid_1Á
lstm_28/while/lstm_cell_28/mulMul(lstm_28/while/lstm_cell_28/Sigmoid_1:y:0lstm_28_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_28/while/lstm_cell_28/mul¨
lstm_28/while/lstm_cell_28/ReluRelu)lstm_28/while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
lstm_28/while/lstm_cell_28/ReluÕ
 lstm_28/while/lstm_cell_28/mul_1Mul&lstm_28/while/lstm_cell_28/Sigmoid:y:0-lstm_28/while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_28/while/lstm_cell_28/mul_1Ê
 lstm_28/while/lstm_cell_28/add_1AddV2"lstm_28/while/lstm_cell_28/mul:z:0$lstm_28/while/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_28/while/lstm_cell_28/add_1µ
$lstm_28/while/lstm_cell_28/Sigmoid_2Sigmoid)lstm_28/while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_28/while/lstm_cell_28/Sigmoid_2§
!lstm_28/while/lstm_cell_28/Relu_1Relu$lstm_28/while/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!lstm_28/while/lstm_cell_28/Relu_1Ù
 lstm_28/while/lstm_cell_28/mul_2Mul(lstm_28/while/lstm_cell_28/Sigmoid_2:y:0/lstm_28/while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_28/while/lstm_cell_28/mul_2
2lstm_28/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_28_while_placeholder_1lstm_28_while_placeholder$lstm_28/while/lstm_cell_28/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_28/while/TensorArrayV2Write/TensorListSetIteml
lstm_28/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_28/while/add/y
lstm_28/while/addAddV2lstm_28_while_placeholderlstm_28/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_28/while/addp
lstm_28/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_28/while/add_1/y
lstm_28/while/add_1AddV2(lstm_28_while_lstm_28_while_loop_counterlstm_28/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_28/while/add_1
lstm_28/while/IdentityIdentitylstm_28/while/add_1:z:0^lstm_28/while/NoOp*
T0*
_output_shapes
: 2
lstm_28/while/Identity¦
lstm_28/while/Identity_1Identity.lstm_28_while_lstm_28_while_maximum_iterations^lstm_28/while/NoOp*
T0*
_output_shapes
: 2
lstm_28/while/Identity_1
lstm_28/while/Identity_2Identitylstm_28/while/add:z:0^lstm_28/while/NoOp*
T0*
_output_shapes
: 2
lstm_28/while/Identity_2º
lstm_28/while/Identity_3IdentityBlstm_28/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_28/while/NoOp*
T0*
_output_shapes
: 2
lstm_28/while/Identity_3®
lstm_28/while/Identity_4Identity$lstm_28/while/lstm_cell_28/mul_2:z:0^lstm_28/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/while/Identity_4®
lstm_28/while/Identity_5Identity$lstm_28/while/lstm_cell_28/add_1:z:0^lstm_28/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/while/Identity_5
lstm_28/while/NoOpNoOp2^lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp1^lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp3^lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_28/while/NoOp"9
lstm_28_while_identitylstm_28/while/Identity:output:0"=
lstm_28_while_identity_1!lstm_28/while/Identity_1:output:0"=
lstm_28_while_identity_2!lstm_28/while/Identity_2:output:0"=
lstm_28_while_identity_3!lstm_28/while/Identity_3:output:0"=
lstm_28_while_identity_4!lstm_28/while/Identity_4:output:0"=
lstm_28_while_identity_5!lstm_28/while/Identity_5:output:0"P
%lstm_28_while_lstm_28_strided_slice_1'lstm_28_while_lstm_28_strided_slice_1_0"z
:lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource<lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource_0"|
;lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource=lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource_0"x
9lstm_28_while_lstm_cell_28_matmul_readvariableop_resource;lstm_28_while_lstm_cell_28_matmul_readvariableop_resource_0"È
alstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensorclstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2f
1lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp1lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp2d
0lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp0lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp2h
2lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp2lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1062456
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1062456___redundant_placeholder05
1while_while_cond_1062456___redundant_placeholder15
1while_while_cond_1062456___redundant_placeholder25
1while_while_cond_1062456___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
ú?

D__inference_lstm_26_layer_call_and_return_conditional_losses_1057993

inputs'
lstm_cell_26_1057911:	¨(
lstm_cell_26_1057913:
ª¨#
lstm_cell_26_1057915:	¨
identity¢$lstm_cell_26/StatefulPartitionedCall¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
$lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_26_1057911lstm_cell_26_1057913lstm_cell_26_1057915*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_10578542&
$lstm_cell_26/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_26_1057911lstm_cell_26_1057913lstm_cell_26_1057915*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1057924*
condR
while_cond_1057923*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

Identity}
NoOpNoOp%^lstm_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_26/StatefulPartitionedCall$lstm_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
f
G__inference_dropout_44_layer_call_and_return_conditional_losses_1063211

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
:ÿÿÿÿÿÿÿÿÿª2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs


I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_1057708

inputs

states
states_11
matmul_readvariableop_resource:	¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates
Ö
f
G__inference_dropout_46_layer_call_and_return_conditional_losses_1063921

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
:ÿÿÿÿÿÿÿÿÿª2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
¯?
Ó
while_body_1059500
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	¨I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	¨G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_26_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	¨*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Ï

è
lstm_26_while_cond_1060999,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3.
*lstm_26_while_less_lstm_26_strided_slice_1E
Alstm_26_while_lstm_26_while_cond_1060999___redundant_placeholder0E
Alstm_26_while_lstm_26_while_cond_1060999___redundant_placeholder1E
Alstm_26_while_lstm_26_while_cond_1060999___redundant_placeholder2E
Alstm_26_while_lstm_26_while_cond_1060999___redundant_placeholder3
lstm_26_while_identity

lstm_26/while/LessLesslstm_26_while_placeholder*lstm_26_while_less_lstm_26_strided_slice_1*
T0*
_output_shapes
: 2
lstm_26/while/Lessu
lstm_26/while/IdentityIdentitylstm_26/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_26/while/Identity"9
lstm_26_while_identitylstm_26/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:

e
G__inference_dropout_45_layer_call_and_return_conditional_losses_1063842

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Å
ø
.__inference_lstm_cell_26_layer_call_fn_1063977

inputs
states_0
states_1
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
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
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_10577082
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1
È
ù
.__inference_lstm_cell_27_layer_call_fn_1064092

inputs
states_0
states_1
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
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
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_10584522
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1
¹
e
,__inference_dropout_44_layer_call_fn_1063194

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
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_44_layer_call_and_return_conditional_losses_10602742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
÷%
î
while_body_1059120
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_28_1059144_0:
ª¨0
while_lstm_cell_28_1059146_0:
ª¨+
while_lstm_cell_28_1059148_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_28_1059144:
ª¨.
while_lstm_cell_28_1059146:
ª¨)
while_lstm_cell_28_1059148:	¨¢*while/lstm_cell_28/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemè
*while/lstm_cell_28/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_28_1059144_0while_lstm_cell_28_1059146_0while_lstm_cell_28_1059148_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_10590502,
*while/lstm_cell_28/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_28/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_28/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_28/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_28/StatefulPartitionedCall*"
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
while_lstm_cell_28_1059144while_lstm_cell_28_1059144_0":
while_lstm_cell_28_1059146while_lstm_cell_28_1059146_0":
while_lstm_cell_28_1059148while_lstm_cell_28_1059148_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2X
*while/lstm_cell_28/StatefulPartitionedCall*while/lstm_cell_28/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
á
º
)__inference_lstm_27_layer_call_fn_1062590
inputs_0
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_27_layer_call_and_return_conditional_losses_10585912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
inputs/0
È
ù
.__inference_lstm_cell_28_layer_call_fn_1064173

inputs
states_0
states_1
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
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
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_10589042
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1
Ö
f
G__inference_dropout_45_layer_call_and_return_conditional_losses_1063854

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
:ÿÿÿÿÿÿÿÿÿª2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
÷
ß
/__inference_sequential_12_layer_call_fn_1060910

inputs
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
	unknown_2:
ª¨
	unknown_3:
ª¨
	unknown_4:	¨
	unknown_5:
ª¨
	unknown_6:
ª¨
	unknown_7:	¨
	unknown_8:
ªª
	unknown_9:	ª

unknown_10:	ª

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
J__inference_sequential_12_layer_call_and_return_conditional_losses_10599942
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
ì 
ý
E__inference_dense_30_layer_call_and_return_conditional_losses_1063960

inputs4
!tensordot_readvariableop_resource:	ª-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	ª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
ú?

D__inference_lstm_26_layer_call_and_return_conditional_losses_1057791

inputs'
lstm_cell_26_1057709:	¨(
lstm_cell_26_1057711:
ª¨#
lstm_cell_26_1057713:	¨
identity¢$lstm_cell_26/StatefulPartitionedCall¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
$lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_26_1057709lstm_cell_26_1057711lstm_cell_26_1057713*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_10577082&
$lstm_cell_26/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_26_1057709lstm_cell_26_1057711lstm_cell_26_1057713*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1057722*
condR
while_cond_1057721*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

Identity}
NoOpNoOp%^lstm_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_26/StatefulPartitionedCall$lstm_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
È
while_cond_1062670
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1062670___redundant_placeholder05
1while_while_cond_1062670___redundant_placeholder15
1while_while_cond_1062670___redundant_placeholder25
1while_while_cond_1062670___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
÷
ß
/__inference_sequential_12_layer_call_fn_1060941

inputs
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
	unknown_2:
ª¨
	unknown_3:
ª¨
	unknown_4:	¨
	unknown_5:
ª¨
	unknown_6:
ª¨
	unknown_7:	¨
	unknown_8:
ªª
	unknown_9:	ª

unknown_10:	ª

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
J__inference_sequential_12_layer_call_and_return_conditional_losses_10607022
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
ò*
Þ
J__inference_sequential_12_layer_call_and_return_conditional_losses_1059994

inputs"
lstm_26_1059585:	¨#
lstm_26_1059587:
ª¨
lstm_26_1059589:	¨#
lstm_27_1059742:
ª¨#
lstm_27_1059744:
ª¨
lstm_27_1059746:	¨#
lstm_28_1059899:
ª¨#
lstm_28_1059901:
ª¨
lstm_28_1059903:	¨$
dense_29_1059945:
ªª
dense_29_1059947:	ª#
dense_30_1059988:	ª
dense_30_1059990:
identity¢ dense_29/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢lstm_26/StatefulPartitionedCall¢lstm_27/StatefulPartitionedCall¢lstm_28/StatefulPartitionedCallª
lstm_26/StatefulPartitionedCallStatefulPartitionedCallinputslstm_26_1059585lstm_26_1059587lstm_26_1059589*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_26_layer_call_and_return_conditional_losses_10595842!
lstm_26/StatefulPartitionedCall
dropout_43/PartitionedCallPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_10595972
dropout_43/PartitionedCallÇ
lstm_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_43/PartitionedCall:output:0lstm_27_1059742lstm_27_1059744lstm_27_1059746*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_27_layer_call_and_return_conditional_losses_10597412!
lstm_27/StatefulPartitionedCall
dropout_44/PartitionedCallPartitionedCall(lstm_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_44_layer_call_and_return_conditional_losses_10597542
dropout_44/PartitionedCallÇ
lstm_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0lstm_28_1059899lstm_28_1059901lstm_28_1059903*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_28_layer_call_and_return_conditional_losses_10598982!
lstm_28/StatefulPartitionedCall
dropout_45/PartitionedCallPartitionedCall(lstm_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_10599112
dropout_45/PartitionedCall¹
 dense_29/StatefulPartitionedCallStatefulPartitionedCall#dropout_45/PartitionedCall:output:0dense_29_1059945dense_29_1059947*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_10599442"
 dense_29/StatefulPartitionedCall
dropout_46/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_46_layer_call_and_return_conditional_losses_10599552
dropout_46/PartitionedCall¸
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_46/PartitionedCall:output:0dense_30_1059988dense_30_1059990*
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
E__inference_dense_30_layer_call_and_return_conditional_losses_10599872"
 dense_30/StatefulPartitionedCall
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityú
NoOpNoOp!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall ^lstm_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall2B
lstm_28/StatefulPartitionedCalllstm_28/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
È
while_cond_1058319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1058319___redundant_placeholder05
1while_while_cond_1058319___redundant_placeholder15
1while_while_cond_1058319___redundant_placeholder25
1while_while_cond_1058319___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
Ö
H
,__inference_dropout_46_layer_call_fn_1063899

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
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_46_layer_call_and_return_conditional_losses_10599552
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
¯?
Ó
while_body_1062171
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	¨I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	¨G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_26_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	¨*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Ï

è
lstm_27_while_cond_1061139,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3.
*lstm_27_while_less_lstm_27_strided_slice_1E
Alstm_27_while_lstm_27_while_cond_1061139___redundant_placeholder0E
Alstm_27_while_lstm_27_while_cond_1061139___redundant_placeholder1E
Alstm_27_while_lstm_27_while_cond_1061139___redundant_placeholder2E
Alstm_27_while_lstm_27_while_cond_1061139___redundant_placeholder3
lstm_27_while_identity

lstm_27/while/LessLesslstm_27_while_placeholder*lstm_27_while_less_lstm_27_strided_slice_1*
T0*
_output_shapes
: 2
lstm_27/while/Lessu
lstm_27/while/IdentityIdentitylstm_27/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_27/while/Identity"9
lstm_27_while_identitylstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
´
·
)__inference_lstm_26_layer_call_fn_1061958

inputs
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_26_layer_call_and_return_conditional_losses_10595842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
Ö
f
G__inference_dropout_45_layer_call_and_return_conditional_losses_1060086

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
:ÿÿÿÿÿÿÿÿÿª2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Þ
È
while_cond_1059499
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1059499___redundant_placeholder05
1while_while_cond_1059499___redundant_placeholder15
1while_while_cond_1059499___redundant_placeholder25
1while_while_cond_1059499___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
³?
Õ
while_body_1059814
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_28_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_28_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_28_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_28_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_28_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_28/BiasAdd/ReadVariableOp¢(while/lstm_cell_28/MatMul/ReadVariableOp¢*while/lstm_cell_28/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_28/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_28_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_28/MatMul/ReadVariableOp×
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMulÐ
*while/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_28_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_28/MatMul_1/ReadVariableOpÀ
while/lstm_cell_28/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMul_1¸
while/lstm_cell_28/addAddV2#while/lstm_cell_28/MatMul:product:0%while/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/addÈ
)while/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_28_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_28/BiasAdd/ReadVariableOpÅ
while/lstm_cell_28/BiasAddBiasAddwhile/lstm_cell_28/add:z:01while/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/BiasAdd
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_28/split/split_dim
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0#while/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_28/split
while/lstm_cell_28/SigmoidSigmoid!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid
while/lstm_cell_28/Sigmoid_1Sigmoid!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_1¡
while/lstm_cell_28/mulMul while/lstm_cell_28/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul
while/lstm_cell_28/ReluRelu!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Reluµ
while/lstm_cell_28/mul_1Mulwhile/lstm_cell_28/Sigmoid:y:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_1ª
while/lstm_cell_28/add_1AddV2while/lstm_cell_28/mul:z:0while/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/add_1
while/lstm_cell_28/Sigmoid_2Sigmoid!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_2
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Relu_1¹
while/lstm_cell_28/mul_2Mul while/lstm_cell_28/Sigmoid_2:y:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_28/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_28/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_28/BiasAdd/ReadVariableOp)^while/lstm_cell_28/MatMul/ReadVariableOp+^while/lstm_cell_28/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_28_biasadd_readvariableop_resource4while_lstm_cell_28_biasadd_readvariableop_resource_0"l
3while_lstm_cell_28_matmul_1_readvariableop_resource5while_lstm_cell_28_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_28_matmul_readvariableop_resource3while_lstm_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_28/BiasAdd/ReadVariableOp)while/lstm_cell_28/BiasAdd/ReadVariableOp2T
(while/lstm_cell_28/MatMul/ReadVariableOp(while/lstm_cell_28/MatMul/ReadVariableOp2X
*while/lstm_cell_28/MatMul_1/ReadVariableOp*while/lstm_cell_28/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Ö
f
G__inference_dropout_43_layer_call_and_return_conditional_losses_1062568

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
:ÿÿÿÿÿÿÿÿÿª2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
¯?
Ó
while_body_1060537
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	¨I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	¨G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_26_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	¨*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
ô

(sequential_12_lstm_28_while_cond_1057501H
Dsequential_12_lstm_28_while_sequential_12_lstm_28_while_loop_counterN
Jsequential_12_lstm_28_while_sequential_12_lstm_28_while_maximum_iterations+
'sequential_12_lstm_28_while_placeholder-
)sequential_12_lstm_28_while_placeholder_1-
)sequential_12_lstm_28_while_placeholder_2-
)sequential_12_lstm_28_while_placeholder_3J
Fsequential_12_lstm_28_while_less_sequential_12_lstm_28_strided_slice_1a
]sequential_12_lstm_28_while_sequential_12_lstm_28_while_cond_1057501___redundant_placeholder0a
]sequential_12_lstm_28_while_sequential_12_lstm_28_while_cond_1057501___redundant_placeholder1a
]sequential_12_lstm_28_while_sequential_12_lstm_28_while_cond_1057501___redundant_placeholder2a
]sequential_12_lstm_28_while_sequential_12_lstm_28_while_cond_1057501___redundant_placeholder3(
$sequential_12_lstm_28_while_identity
Þ
 sequential_12/lstm_28/while/LessLess'sequential_12_lstm_28_while_placeholderFsequential_12_lstm_28_while_less_sequential_12_lstm_28_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_12/lstm_28/while/Less
$sequential_12/lstm_28/while/IdentityIdentity$sequential_12/lstm_28/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_12/lstm_28/while/Identity"U
$sequential_12_lstm_28_while_identity-sequential_12/lstm_28/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
ã^

(sequential_12_lstm_26_while_body_1057222H
Dsequential_12_lstm_26_while_sequential_12_lstm_26_while_loop_counterN
Jsequential_12_lstm_26_while_sequential_12_lstm_26_while_maximum_iterations+
'sequential_12_lstm_26_while_placeholder-
)sequential_12_lstm_26_while_placeholder_1-
)sequential_12_lstm_26_while_placeholder_2-
)sequential_12_lstm_26_while_placeholder_3G
Csequential_12_lstm_26_while_sequential_12_lstm_26_strided_slice_1_0
sequential_12_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_26_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_12_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	¨_
Ksequential_12_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ª¨Y
Jsequential_12_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	¨(
$sequential_12_lstm_26_while_identity*
&sequential_12_lstm_26_while_identity_1*
&sequential_12_lstm_26_while_identity_2*
&sequential_12_lstm_26_while_identity_3*
&sequential_12_lstm_26_while_identity_4*
&sequential_12_lstm_26_while_identity_5E
Asequential_12_lstm_26_while_sequential_12_lstm_26_strided_slice_1
}sequential_12_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_26_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_12_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	¨]
Isequential_12_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨W
Hsequential_12_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	¨¢?sequential_12/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp¢>sequential_12/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp¢@sequential_12/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpï
Msequential_12/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2O
Msequential_12/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential_12/lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_26_tensorarrayunstack_tensorlistfromtensor_0'sequential_12_lstm_26_while_placeholderVsequential_12/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential_12/lstm_26/while/TensorArrayV2Read/TensorListGetItem
>sequential_12/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOpIsequential_12_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	¨*
dtype02@
>sequential_12/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp¯
/sequential_12/lstm_26/while/lstm_cell_26/MatMulMatMulFsequential_12/lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨21
/sequential_12/lstm_26/while/lstm_cell_26/MatMul
@sequential_12/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02B
@sequential_12/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp
1sequential_12/lstm_26/while/lstm_cell_26/MatMul_1MatMul)sequential_12_lstm_26_while_placeholder_2Hsequential_12/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨23
1sequential_12/lstm_26/while/lstm_cell_26/MatMul_1
,sequential_12/lstm_26/while/lstm_cell_26/addAddV29sequential_12/lstm_26/while/lstm_cell_26/MatMul:product:0;sequential_12/lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2.
,sequential_12/lstm_26/while/lstm_cell_26/add
?sequential_12/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02A
?sequential_12/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp
0sequential_12/lstm_26/while/lstm_cell_26/BiasAddBiasAdd0sequential_12/lstm_26/while/lstm_cell_26/add:z:0Gsequential_12/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨22
0sequential_12/lstm_26/while/lstm_cell_26/BiasAdd¶
8sequential_12/lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_26/while/lstm_cell_26/split/split_dimç
.sequential_12/lstm_26/while/lstm_cell_26/splitSplitAsequential_12/lstm_26/while/lstm_cell_26/split/split_dim:output:09sequential_12/lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split20
.sequential_12/lstm_26/while/lstm_cell_26/splitÛ
0sequential_12/lstm_26/while/lstm_cell_26/SigmoidSigmoid7sequential_12/lstm_26/while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª22
0sequential_12/lstm_26/while/lstm_cell_26/Sigmoidß
2sequential_12/lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid7sequential_12/lstm_26/while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª24
2sequential_12/lstm_26/while/lstm_cell_26/Sigmoid_1ù
,sequential_12/lstm_26/while/lstm_cell_26/mulMul6sequential_12/lstm_26/while/lstm_cell_26/Sigmoid_1:y:0)sequential_12_lstm_26_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2.
,sequential_12/lstm_26/while/lstm_cell_26/mulÒ
-sequential_12/lstm_26/while/lstm_cell_26/ReluRelu7sequential_12/lstm_26/while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2/
-sequential_12/lstm_26/while/lstm_cell_26/Relu
.sequential_12/lstm_26/while/lstm_cell_26/mul_1Mul4sequential_12/lstm_26/while/lstm_cell_26/Sigmoid:y:0;sequential_12/lstm_26/while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª20
.sequential_12/lstm_26/while/lstm_cell_26/mul_1
.sequential_12/lstm_26/while/lstm_cell_26/add_1AddV20sequential_12/lstm_26/while/lstm_cell_26/mul:z:02sequential_12/lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª20
.sequential_12/lstm_26/while/lstm_cell_26/add_1ß
2sequential_12/lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid7sequential_12/lstm_26/while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª24
2sequential_12/lstm_26/while/lstm_cell_26/Sigmoid_2Ñ
/sequential_12/lstm_26/while/lstm_cell_26/Relu_1Relu2sequential_12/lstm_26/while/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª21
/sequential_12/lstm_26/while/lstm_cell_26/Relu_1
.sequential_12/lstm_26/while/lstm_cell_26/mul_2Mul6sequential_12/lstm_26/while/lstm_cell_26/Sigmoid_2:y:0=sequential_12/lstm_26/while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª20
.sequential_12/lstm_26/while/lstm_cell_26/mul_2Î
@sequential_12/lstm_26/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_12_lstm_26_while_placeholder_1'sequential_12_lstm_26_while_placeholder2sequential_12/lstm_26/while/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_12/lstm_26/while/TensorArrayV2Write/TensorListSetItem
!sequential_12/lstm_26/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_12/lstm_26/while/add/yÁ
sequential_12/lstm_26/while/addAddV2'sequential_12_lstm_26_while_placeholder*sequential_12/lstm_26/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_26/while/add
#sequential_12/lstm_26/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_12/lstm_26/while/add_1/yä
!sequential_12/lstm_26/while/add_1AddV2Dsequential_12_lstm_26_while_sequential_12_lstm_26_while_loop_counter,sequential_12/lstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_26/while/add_1Ã
$sequential_12/lstm_26/while/IdentityIdentity%sequential_12/lstm_26/while/add_1:z:0!^sequential_12/lstm_26/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_12/lstm_26/while/Identityì
&sequential_12/lstm_26/while/Identity_1IdentityJsequential_12_lstm_26_while_sequential_12_lstm_26_while_maximum_iterations!^sequential_12/lstm_26/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_26/while/Identity_1Å
&sequential_12/lstm_26/while/Identity_2Identity#sequential_12/lstm_26/while/add:z:0!^sequential_12/lstm_26/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_26/while/Identity_2ò
&sequential_12/lstm_26/while/Identity_3IdentityPsequential_12/lstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_12/lstm_26/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_26/while/Identity_3æ
&sequential_12/lstm_26/while/Identity_4Identity2sequential_12/lstm_26/while/lstm_cell_26/mul_2:z:0!^sequential_12/lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2(
&sequential_12/lstm_26/while/Identity_4æ
&sequential_12/lstm_26/while/Identity_5Identity2sequential_12/lstm_26/while/lstm_cell_26/add_1:z:0!^sequential_12/lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2(
&sequential_12/lstm_26/while/Identity_5Ì
 sequential_12/lstm_26/while/NoOpNoOp@^sequential_12/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp?^sequential_12/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpA^sequential_12/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_12/lstm_26/while/NoOp"U
$sequential_12_lstm_26_while_identity-sequential_12/lstm_26/while/Identity:output:0"Y
&sequential_12_lstm_26_while_identity_1/sequential_12/lstm_26/while/Identity_1:output:0"Y
&sequential_12_lstm_26_while_identity_2/sequential_12/lstm_26/while/Identity_2:output:0"Y
&sequential_12_lstm_26_while_identity_3/sequential_12/lstm_26/while/Identity_3:output:0"Y
&sequential_12_lstm_26_while_identity_4/sequential_12/lstm_26/while/Identity_4:output:0"Y
&sequential_12_lstm_26_while_identity_5/sequential_12/lstm_26/while/Identity_5:output:0"
Hsequential_12_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resourceJsequential_12_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0"
Isequential_12_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resourceKsequential_12_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0"
Gsequential_12_lstm_26_while_lstm_cell_26_matmul_readvariableop_resourceIsequential_12_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"
Asequential_12_lstm_26_while_sequential_12_lstm_26_strided_slice_1Csequential_12_lstm_26_while_sequential_12_lstm_26_strided_slice_1_0"
}sequential_12_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_26_tensorarrayunstack_tensorlistfromtensorsequential_12_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2
?sequential_12/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp?sequential_12/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp2
>sequential_12/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp>sequential_12/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp2
@sequential_12/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp@sequential_12/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1059656
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1059656___redundant_placeholder05
1while_while_cond_1059656___redundant_placeholder15
1while_while_cond_1059656___redundant_placeholder25
1while_while_cond_1059656___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
½U

D__inference_lstm_26_layer_call_and_return_conditional_losses_1059584

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	¨A
-lstm_cell_26_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_26_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1059500*
condR
while_cond_1059499*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
¸
)__inference_lstm_28_layer_call_fn_1063244

inputs
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_28_layer_call_and_return_conditional_losses_10598982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
ÃU

D__inference_lstm_27_layer_call_and_return_conditional_losses_1063041

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ª¨A
-lstm_cell_27_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_27_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
:ÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1062957*
condR
while_cond_1062956*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
V
 
D__inference_lstm_28_layer_call_and_return_conditional_losses_1063541
inputs_0?
+lstm_cell_28_matmul_readvariableop_resource:
ª¨A
-lstm_cell_28_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_28_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_28/BiasAdd/ReadVariableOp¢"lstm_cell_28/MatMul/ReadVariableOp¢$lstm_cell_28/MatMul_1/ReadVariableOp¢whileF
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_28/MatMul/ReadVariableOpReadVariableOp+lstm_cell_28_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_28/MatMul/ReadVariableOp­
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0*lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul¼
$lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_28_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_28/MatMul_1/ReadVariableOp©
lstm_cell_28/MatMul_1MatMulzeros:output:0,lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul_1 
lstm_cell_28/addAddV2lstm_cell_28/MatMul:product:0lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/add´
#lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_28_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_28/BiasAdd/ReadVariableOp­
lstm_cell_28/BiasAddBiasAddlstm_cell_28/add:z:0+lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/BiasAdd~
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_28/split/split_dim÷
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_28/split
lstm_cell_28/SigmoidSigmoidlstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid
lstm_cell_28/Sigmoid_1Sigmoidlstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_1
lstm_cell_28/mulMullstm_cell_28/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul~
lstm_cell_28/ReluRelulstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu
lstm_cell_28/mul_1Mullstm_cell_28/Sigmoid:y:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_1
lstm_cell_28/add_1AddV2lstm_cell_28/mul:z:0lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/add_1
lstm_cell_28/Sigmoid_2Sigmoidlstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_2}
lstm_cell_28/Relu_1Relulstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu_1¡
lstm_cell_28/mul_2Mullstm_cell_28/Sigmoid_2:y:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_28_matmul_readvariableop_resource-lstm_cell_28_matmul_1_readvariableop_resource,lstm_cell_28_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1063457*
condR
while_cond_1063456*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_28/BiasAdd/ReadVariableOp#^lstm_cell_28/MatMul/ReadVariableOp%^lstm_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_28/BiasAdd/ReadVariableOp#lstm_cell_28/BiasAdd/ReadVariableOp2H
"lstm_cell_28/MatMul/ReadVariableOp"lstm_cell_28/MatMul/ReadVariableOp2L
$lstm_cell_28/MatMul_1/ReadVariableOp$lstm_cell_28/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
inputs/0

e
G__inference_dropout_45_layer_call_and_return_conditional_losses_1059911

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs


*__inference_dense_29_layer_call_fn_1063863

inputs
unknown:
ªª
	unknown_0:	ª
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_10599442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿª: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Å
ø
.__inference_lstm_cell_26_layer_call_fn_1063994

inputs
states_0
states_1
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
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
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_10578542
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1

e
G__inference_dropout_46_layer_call_and_return_conditional_losses_1063909

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
¯?
Ó
while_body_1062314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	¨I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	¨G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_26_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	¨*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
³?
Õ
while_body_1062671
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_27_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
ãÑ
Ö 
#__inference__traced_restore_1064575
file_prefix4
 assignvariableop_dense_29_kernel:
ªª/
 assignvariableop_1_dense_29_bias:	ª5
"assignvariableop_2_dense_30_kernel:	ª.
 assignvariableop_3_dense_30_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: A
.assignvariableop_9_lstm_26_lstm_cell_26_kernel:	¨M
9assignvariableop_10_lstm_26_lstm_cell_26_recurrent_kernel:
ª¨<
-assignvariableop_11_lstm_26_lstm_cell_26_bias:	¨C
/assignvariableop_12_lstm_27_lstm_cell_27_kernel:
ª¨M
9assignvariableop_13_lstm_27_lstm_cell_27_recurrent_kernel:
ª¨<
-assignvariableop_14_lstm_27_lstm_cell_27_bias:	¨C
/assignvariableop_15_lstm_28_lstm_cell_28_kernel:
ª¨M
9assignvariableop_16_lstm_28_lstm_cell_28_recurrent_kernel:
ª¨<
-assignvariableop_17_lstm_28_lstm_cell_28_bias:	¨#
assignvariableop_18_total: #
assignvariableop_19_count: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: >
*assignvariableop_22_adam_dense_29_kernel_m:
ªª7
(assignvariableop_23_adam_dense_29_bias_m:	ª=
*assignvariableop_24_adam_dense_30_kernel_m:	ª6
(assignvariableop_25_adam_dense_30_bias_m:I
6assignvariableop_26_adam_lstm_26_lstm_cell_26_kernel_m:	¨T
@assignvariableop_27_adam_lstm_26_lstm_cell_26_recurrent_kernel_m:
ª¨C
4assignvariableop_28_adam_lstm_26_lstm_cell_26_bias_m:	¨J
6assignvariableop_29_adam_lstm_27_lstm_cell_27_kernel_m:
ª¨T
@assignvariableop_30_adam_lstm_27_lstm_cell_27_recurrent_kernel_m:
ª¨C
4assignvariableop_31_adam_lstm_27_lstm_cell_27_bias_m:	¨J
6assignvariableop_32_adam_lstm_28_lstm_cell_28_kernel_m:
ª¨T
@assignvariableop_33_adam_lstm_28_lstm_cell_28_recurrent_kernel_m:
ª¨C
4assignvariableop_34_adam_lstm_28_lstm_cell_28_bias_m:	¨>
*assignvariableop_35_adam_dense_29_kernel_v:
ªª7
(assignvariableop_36_adam_dense_29_bias_v:	ª=
*assignvariableop_37_adam_dense_30_kernel_v:	ª6
(assignvariableop_38_adam_dense_30_bias_v:I
6assignvariableop_39_adam_lstm_26_lstm_cell_26_kernel_v:	¨T
@assignvariableop_40_adam_lstm_26_lstm_cell_26_recurrent_kernel_v:
ª¨C
4assignvariableop_41_adam_lstm_26_lstm_cell_26_bias_v:	¨J
6assignvariableop_42_adam_lstm_27_lstm_cell_27_kernel_v:
ª¨T
@assignvariableop_43_adam_lstm_27_lstm_cell_27_recurrent_kernel_v:
ª¨C
4assignvariableop_44_adam_lstm_27_lstm_cell_27_bias_v:	¨J
6assignvariableop_45_adam_lstm_28_lstm_cell_28_kernel_v:
ª¨T
@assignvariableop_46_adam_lstm_28_lstm_cell_28_recurrent_kernel_v:
ª¨C
4assignvariableop_47_adam_lstm_28_lstm_cell_28_bias_v:	¨
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
AssignVariableOpAssignVariableOp assignvariableop_dense_29_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_29_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_30_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_30_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_26_lstm_cell_26_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Á
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_26_lstm_cell_26_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11µ
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_26_lstm_cell_26_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_lstm_27_lstm_cell_27_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Á
AssignVariableOp_13AssignVariableOp9assignvariableop_13_lstm_27_lstm_cell_27_recurrent_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14µ
AssignVariableOp_14AssignVariableOp-assignvariableop_14_lstm_27_lstm_cell_27_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15·
AssignVariableOp_15AssignVariableOp/assignvariableop_15_lstm_28_lstm_cell_28_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Á
AssignVariableOp_16AssignVariableOp9assignvariableop_16_lstm_28_lstm_cell_28_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17µ
AssignVariableOp_17AssignVariableOp-assignvariableop_17_lstm_28_lstm_cell_28_biasIdentity_17:output:0"/device:CPU:0*
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
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_29_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23°
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_29_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_30_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25°
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_dense_30_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¾
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_lstm_26_lstm_cell_26_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27È
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_lstm_26_lstm_cell_26_recurrent_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¼
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_lstm_26_lstm_cell_26_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¾
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_lstm_27_lstm_cell_27_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30È
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_lstm_27_lstm_cell_27_recurrent_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¼
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_lstm_27_lstm_cell_27_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¾
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_lstm_28_lstm_cell_28_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33È
AssignVariableOp_33AssignVariableOp@assignvariableop_33_adam_lstm_28_lstm_cell_28_recurrent_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¼
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_lstm_28_lstm_cell_28_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_29_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36°
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_29_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_30_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_30_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¾
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_lstm_26_lstm_cell_26_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40È
AssignVariableOp_40AssignVariableOp@assignvariableop_40_adam_lstm_26_lstm_cell_26_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¼
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_lstm_26_lstm_cell_26_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¾
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_lstm_27_lstm_cell_27_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43È
AssignVariableOp_43AssignVariableOp@assignvariableop_43_adam_lstm_27_lstm_cell_27_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¼
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_lstm_27_lstm_cell_27_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¾
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_lstm_28_lstm_cell_28_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46È
AssignVariableOp_46AssignVariableOp@assignvariableop_46_adam_lstm_28_lstm_cell_28_recurrent_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¼
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_lstm_28_lstm_cell_28_bias_vIdentity_47:output:0"/device:CPU:0*
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
Ö
H
,__inference_dropout_43_layer_call_fn_1062546

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
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_10595972
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs


I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_1064254

inputs
states_0
states_12
matmul_readvariableop_resource:
ª¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1
Þ
È
while_cond_1063313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1063313___redundant_placeholder05
1while_while_cond_1063313___redundant_placeholder15
1while_while_cond_1063313___redundant_placeholder25
1while_while_cond_1063313___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
ÿ?

D__inference_lstm_27_layer_call_and_return_conditional_losses_1058389

inputs(
lstm_cell_27_1058307:
ª¨(
lstm_cell_27_1058309:
ª¨#
lstm_cell_27_1058311:	¨
identity¢$lstm_cell_27/StatefulPartitionedCall¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¤
$lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_27_1058307lstm_cell_27_1058309lstm_cell_27_1058311*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_10583062&
$lstm_cell_27/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_27_1058307lstm_cell_27_1058309lstm_cell_27_1058311*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1058320*
condR
while_cond_1058319*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

Identity}
NoOpNoOp%^lstm_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 2L
$lstm_cell_27/StatefulPartitionedCall$lstm_cell_27/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs


I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_1058452

inputs

states
states_12
matmul_readvariableop_resource:
ª¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates
±1
ò
J__inference_sequential_12_layer_call_and_return_conditional_losses_1060702

inputs"
lstm_26_1060666:	¨#
lstm_26_1060668:
ª¨
lstm_26_1060670:	¨#
lstm_27_1060674:
ª¨#
lstm_27_1060676:
ª¨
lstm_27_1060678:	¨#
lstm_28_1060682:
ª¨#
lstm_28_1060684:
ª¨
lstm_28_1060686:	¨$
dense_29_1060690:
ªª
dense_29_1060692:	ª#
dense_30_1060696:	ª
dense_30_1060698:
identity¢ dense_29/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢"dropout_43/StatefulPartitionedCall¢"dropout_44/StatefulPartitionedCall¢"dropout_45/StatefulPartitionedCall¢"dropout_46/StatefulPartitionedCall¢lstm_26/StatefulPartitionedCall¢lstm_27/StatefulPartitionedCall¢lstm_28/StatefulPartitionedCallª
lstm_26/StatefulPartitionedCallStatefulPartitionedCallinputslstm_26_1060666lstm_26_1060668lstm_26_1060670*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_26_layer_call_and_return_conditional_losses_10606212!
lstm_26/StatefulPartitionedCall
"dropout_43/StatefulPartitionedCallStatefulPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_10604622$
"dropout_43/StatefulPartitionedCallÏ
lstm_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_43/StatefulPartitionedCall:output:0lstm_27_1060674lstm_27_1060676lstm_27_1060678*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_27_layer_call_and_return_conditional_losses_10604332!
lstm_27/StatefulPartitionedCall¿
"dropout_44/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0#^dropout_43/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_44_layer_call_and_return_conditional_losses_10602742$
"dropout_44/StatefulPartitionedCallÏ
lstm_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_44/StatefulPartitionedCall:output:0lstm_28_1060682lstm_28_1060684lstm_28_1060686*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_28_layer_call_and_return_conditional_losses_10602452!
lstm_28/StatefulPartitionedCall¿
"dropout_45/StatefulPartitionedCallStatefulPartitionedCall(lstm_28/StatefulPartitionedCall:output:0#^dropout_44/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_10600862$
"dropout_45/StatefulPartitionedCallÁ
 dense_29/StatefulPartitionedCallStatefulPartitionedCall+dropout_45/StatefulPartitionedCall:output:0dense_29_1060690dense_29_1060692*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_10599442"
 dense_29/StatefulPartitionedCallÀ
"dropout_46/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0#^dropout_45/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_46_layer_call_and_return_conditional_losses_10600532$
"dropout_46/StatefulPartitionedCallÀ
 dense_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_46/StatefulPartitionedCall:output:0dense_30_1060696dense_30_1060698*
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
E__inference_dense_30_layer_call_and_return_conditional_losses_10599872"
 dense_30/StatefulPartitionedCall
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall#^dropout_43/StatefulPartitionedCall#^dropout_44/StatefulPartitionedCall#^dropout_45/StatefulPartitionedCall#^dropout_46/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall ^lstm_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2H
"dropout_43/StatefulPartitionedCall"dropout_43/StatefulPartitionedCall2H
"dropout_44/StatefulPartitionedCall"dropout_44/StatefulPartitionedCall2H
"dropout_45/StatefulPartitionedCall"dropout_45/StatefulPartitionedCall2H
"dropout_46/StatefulPartitionedCall"dropout_46/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall2B
lstm_28/StatefulPartitionedCalllstm_28/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷%
î
while_body_1058522
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_27_1058546_0:
ª¨0
while_lstm_cell_27_1058548_0:
ª¨+
while_lstm_cell_27_1058550_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_27_1058546:
ª¨.
while_lstm_cell_27_1058548:
ª¨)
while_lstm_cell_27_1058550:	¨¢*while/lstm_cell_27/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemè
*while/lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_27_1058546_0while_lstm_cell_27_1058548_0while_lstm_cell_27_1058550_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_10584522,
*while/lstm_cell_27/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_27/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_27/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_27/StatefulPartitionedCall*"
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
while_lstm_cell_27_1058546while_lstm_cell_27_1058546_0":
while_lstm_cell_27_1058548while_lstm_cell_27_1058548_0":
while_lstm_cell_27_1058550while_lstm_cell_27_1058550_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2X
*while/lstm_cell_27/StatefulPartitionedCall*while/lstm_cell_27/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
ô%
ì
while_body_1057924
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_26_1057948_0:	¨0
while_lstm_cell_26_1057950_0:
ª¨+
while_lstm_cell_26_1057952_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_26_1057948:	¨.
while_lstm_cell_26_1057950:
ª¨)
while_lstm_cell_26_1057952:	¨¢*while/lstm_cell_26/StatefulPartitionedCallÃ
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
*while/lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_26_1057948_0while_lstm_cell_26_1057950_0while_lstm_cell_26_1057952_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_10578542,
*while/lstm_cell_26/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_26/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_26/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_26/StatefulPartitionedCall*"
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
while_lstm_cell_26_1057948while_lstm_cell_26_1057948_0":
while_lstm_cell_26_1057950while_lstm_cell_26_1057950_0":
while_lstm_cell_26_1057952while_lstm_cell_26_1057952_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2X
*while/lstm_cell_26/StatefulPartitionedCall*while/lstm_cell_26/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
á
º
)__inference_lstm_28_layer_call_fn_1063222
inputs_0
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_28_layer_call_and_return_conditional_losses_10589872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
inputs/0
ÃU

D__inference_lstm_27_layer_call_and_return_conditional_losses_1060433

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ª¨A
-lstm_cell_27_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_27_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
:ÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1060349*
condR
while_cond_1060348*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs


I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_1058306

inputs

states
states_12
matmul_readvariableop_resource:
ª¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
B:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_namestates

e
G__inference_dropout_43_layer_call_and_return_conditional_losses_1062556

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs


I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_1064026

inputs
states_0
states_11
matmul_readvariableop_resource:	¨4
 matmul_1_readvariableop_resource:
ª¨.
biasadd_readvariableop_resource:	¨
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2	
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
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
A:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
"
_user_specified_name
states/1
Ï

è
lstm_28_while_cond_1061771,
(lstm_28_while_lstm_28_while_loop_counter2
.lstm_28_while_lstm_28_while_maximum_iterations
lstm_28_while_placeholder
lstm_28_while_placeholder_1
lstm_28_while_placeholder_2
lstm_28_while_placeholder_3.
*lstm_28_while_less_lstm_28_strided_slice_1E
Alstm_28_while_lstm_28_while_cond_1061771___redundant_placeholder0E
Alstm_28_while_lstm_28_while_cond_1061771___redundant_placeholder1E
Alstm_28_while_lstm_28_while_cond_1061771___redundant_placeholder2E
Alstm_28_while_lstm_28_while_cond_1061771___redundant_placeholder3
lstm_28_while_identity

lstm_28/while/LessLesslstm_28_while_placeholder*lstm_28_while_less_lstm_28_strided_slice_1*
T0*
_output_shapes
: 2
lstm_28/while/Lessu
lstm_28/while/IdentityIdentitylstm_28/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_28/while/Identity"9
lstm_28_while_identitylstm_28/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
³?
Õ
while_body_1060349
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_27_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1062313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1062313___redundant_placeholder05
1while_while_cond_1062313___redundant_placeholder15
1while_while_cond_1062313___redundant_placeholder25
1while_while_cond_1062313___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
Ö
H
,__inference_dropout_44_layer_call_fn_1063189

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
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_44_layer_call_and_return_conditional_losses_10597542
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Ö
H
,__inference_dropout_45_layer_call_fn_1063832

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
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_10599112
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
äJ
Ó

lstm_26_while_body_1061000,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3+
'lstm_26_while_lstm_26_strided_slice_1_0g
clstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	¨Q
=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ª¨K
<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	¨
lstm_26_while_identity
lstm_26_while_identity_1
lstm_26_while_identity_2
lstm_26_while_identity_3
lstm_26_while_identity_4
lstm_26_while_identity_5)
%lstm_26_while_lstm_26_strided_slice_1e
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorL
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	¨O
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨I
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	¨¢1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp¢0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp¢2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpÓ
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0lstm_26_while_placeholderHlstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_26/while/TensorArrayV2Read/TensorListGetItemá
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	¨*
dtype022
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp÷
!lstm_26/while/lstm_cell_26/MatMulMatMul8lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2#
!lstm_26/while/lstm_cell_26/MatMulè
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype024
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpà
#lstm_26/while/lstm_cell_26/MatMul_1MatMullstm_26_while_placeholder_2:lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2%
#lstm_26/while/lstm_cell_26/MatMul_1Ø
lstm_26/while/lstm_cell_26/addAddV2+lstm_26/while/lstm_cell_26/MatMul:product:0-lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2 
lstm_26/while/lstm_cell_26/addà
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype023
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpå
"lstm_26/while/lstm_cell_26/BiasAddBiasAdd"lstm_26/while/lstm_cell_26/add:z:09lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2$
"lstm_26/while/lstm_cell_26/BiasAdd
*lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_26/while/lstm_cell_26/split/split_dim¯
 lstm_26/while/lstm_cell_26/splitSplit3lstm_26/while/lstm_cell_26/split/split_dim:output:0+lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2"
 lstm_26/while/lstm_cell_26/split±
"lstm_26/while/lstm_cell_26/SigmoidSigmoid)lstm_26/while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2$
"lstm_26/while/lstm_cell_26/Sigmoidµ
$lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid)lstm_26/while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_26/while/lstm_cell_26/Sigmoid_1Á
lstm_26/while/lstm_cell_26/mulMul(lstm_26/while/lstm_cell_26/Sigmoid_1:y:0lstm_26_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_26/while/lstm_cell_26/mul¨
lstm_26/while/lstm_cell_26/ReluRelu)lstm_26/while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
lstm_26/while/lstm_cell_26/ReluÕ
 lstm_26/while/lstm_cell_26/mul_1Mul&lstm_26/while/lstm_cell_26/Sigmoid:y:0-lstm_26/while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_26/while/lstm_cell_26/mul_1Ê
 lstm_26/while/lstm_cell_26/add_1AddV2"lstm_26/while/lstm_cell_26/mul:z:0$lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_26/while/lstm_cell_26/add_1µ
$lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid)lstm_26/while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_26/while/lstm_cell_26/Sigmoid_2§
!lstm_26/while/lstm_cell_26/Relu_1Relu$lstm_26/while/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!lstm_26/while/lstm_cell_26/Relu_1Ù
 lstm_26/while/lstm_cell_26/mul_2Mul(lstm_26/while/lstm_cell_26/Sigmoid_2:y:0/lstm_26/while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_26/while/lstm_cell_26/mul_2
2lstm_26/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_26_while_placeholder_1lstm_26_while_placeholder$lstm_26/while/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_26/while/TensorArrayV2Write/TensorListSetIteml
lstm_26/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_26/while/add/y
lstm_26/while/addAddV2lstm_26_while_placeholderlstm_26/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_26/while/addp
lstm_26/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_26/while/add_1/y
lstm_26/while/add_1AddV2(lstm_26_while_lstm_26_while_loop_counterlstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_26/while/add_1
lstm_26/while/IdentityIdentitylstm_26/while/add_1:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity¦
lstm_26/while/Identity_1Identity.lstm_26_while_lstm_26_while_maximum_iterations^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_1
lstm_26/while/Identity_2Identitylstm_26/while/add:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_2º
lstm_26/while/Identity_3IdentityBlstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_3®
lstm_26/while/Identity_4Identity$lstm_26/while/lstm_cell_26/mul_2:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/while/Identity_4®
lstm_26/while/Identity_5Identity$lstm_26/while/lstm_cell_26/add_1:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/while/Identity_5
lstm_26/while/NoOpNoOp2^lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1^lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp3^lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_26/while/NoOp"9
lstm_26_while_identitylstm_26/while/Identity:output:0"=
lstm_26_while_identity_1!lstm_26/while/Identity_1:output:0"=
lstm_26_while_identity_2!lstm_26/while/Identity_2:output:0"=
lstm_26_while_identity_3!lstm_26/while/Identity_3:output:0"=
lstm_26_while_identity_4!lstm_26/while/Identity_4:output:0"=
lstm_26_while_identity_5!lstm_26/while/Identity_5:output:0"P
%lstm_26_while_lstm_26_strided_slice_1'lstm_26_while_lstm_26_strided_slice_1_0"z
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0"|
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0"x
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"È
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2f
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp2d
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp2h
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
½U

D__inference_lstm_26_layer_call_and_return_conditional_losses_1062541

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	¨A
-lstm_cell_26_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_26_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1062457*
condR
while_cond_1062456*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯?
Ó
while_body_1062028
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	¨I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	¨G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_26_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	¨*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
üU

D__inference_lstm_26_layer_call_and_return_conditional_losses_1062255
inputs_0>
+lstm_cell_26_matmul_readvariableop_resource:	¨A
-lstm_cell_26_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_26_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileF
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1062171*
condR
while_cond_1062170*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÃU

D__inference_lstm_27_layer_call_and_return_conditional_losses_1059741

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ª¨A
-lstm_cell_27_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_27_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
:ÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1059657*
condR
while_cond_1059656*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Þ
È
while_cond_1063599
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1063599___redundant_placeholder05
1while_while_cond_1063599___redundant_placeholder15
1while_while_cond_1063599___redundant_placeholder25
1while_while_cond_1063599___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
÷%
î
while_body_1058918
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_28_1058942_0:
ª¨0
while_lstm_cell_28_1058944_0:
ª¨+
while_lstm_cell_28_1058946_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_28_1058942:
ª¨.
while_lstm_cell_28_1058944:
ª¨)
while_lstm_cell_28_1058946:	¨¢*while/lstm_cell_28/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemè
*while/lstm_cell_28/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_28_1058942_0while_lstm_cell_28_1058944_0while_lstm_cell_28_1058946_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_10589042,
*while/lstm_cell_28/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_28/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_28/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_28/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_28/StatefulPartitionedCall*"
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
while_lstm_cell_28_1058942while_lstm_cell_28_1058942_0":
while_lstm_cell_28_1058944while_lstm_cell_28_1058944_0":
while_lstm_cell_28_1058946while_lstm_cell_28_1058946_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2X
*while/lstm_cell_28/StatefulPartitionedCall*while/lstm_cell_28/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
·
¸
)__inference_lstm_27_layer_call_fn_1062612

inputs
unknown:
ª¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_27_layer_call_and_return_conditional_losses_10604332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
èJ
Õ

lstm_28_while_body_1061280,
(lstm_28_while_lstm_28_while_loop_counter2
.lstm_28_while_lstm_28_while_maximum_iterations
lstm_28_while_placeholder
lstm_28_while_placeholder_1
lstm_28_while_placeholder_2
lstm_28_while_placeholder_3+
'lstm_28_while_lstm_28_strided_slice_1_0g
clstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_28_while_lstm_cell_28_matmul_readvariableop_resource_0:
ª¨Q
=lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource_0:
ª¨K
<lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource_0:	¨
lstm_28_while_identity
lstm_28_while_identity_1
lstm_28_while_identity_2
lstm_28_while_identity_3
lstm_28_while_identity_4
lstm_28_while_identity_5)
%lstm_28_while_lstm_28_strided_slice_1e
alstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensorM
9lstm_28_while_lstm_cell_28_matmul_readvariableop_resource:
ª¨O
;lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨I
:lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource:	¨¢1lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp¢0lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp¢2lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOpÓ
?lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2A
?lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_28/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0lstm_28_while_placeholderHlstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype023
1lstm_28/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_28/while/lstm_cell_28/MatMul/ReadVariableOpReadVariableOp;lstm_28_while_lstm_cell_28_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype022
0lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp÷
!lstm_28/while/lstm_cell_28/MatMulMatMul8lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2#
!lstm_28/while/lstm_cell_28/MatMulè
2lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp=lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype024
2lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOpà
#lstm_28/while/lstm_cell_28/MatMul_1MatMullstm_28_while_placeholder_2:lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2%
#lstm_28/while/lstm_cell_28/MatMul_1Ø
lstm_28/while/lstm_cell_28/addAddV2+lstm_28/while/lstm_cell_28/MatMul:product:0-lstm_28/while/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2 
lstm_28/while/lstm_cell_28/addà
1lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp<lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype023
1lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOpå
"lstm_28/while/lstm_cell_28/BiasAddBiasAdd"lstm_28/while/lstm_cell_28/add:z:09lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2$
"lstm_28/while/lstm_cell_28/BiasAdd
*lstm_28/while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_28/while/lstm_cell_28/split/split_dim¯
 lstm_28/while/lstm_cell_28/splitSplit3lstm_28/while/lstm_cell_28/split/split_dim:output:0+lstm_28/while/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2"
 lstm_28/while/lstm_cell_28/split±
"lstm_28/while/lstm_cell_28/SigmoidSigmoid)lstm_28/while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2$
"lstm_28/while/lstm_cell_28/Sigmoidµ
$lstm_28/while/lstm_cell_28/Sigmoid_1Sigmoid)lstm_28/while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_28/while/lstm_cell_28/Sigmoid_1Á
lstm_28/while/lstm_cell_28/mulMul(lstm_28/while/lstm_cell_28/Sigmoid_1:y:0lstm_28_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_28/while/lstm_cell_28/mul¨
lstm_28/while/lstm_cell_28/ReluRelu)lstm_28/while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
lstm_28/while/lstm_cell_28/ReluÕ
 lstm_28/while/lstm_cell_28/mul_1Mul&lstm_28/while/lstm_cell_28/Sigmoid:y:0-lstm_28/while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_28/while/lstm_cell_28/mul_1Ê
 lstm_28/while/lstm_cell_28/add_1AddV2"lstm_28/while/lstm_cell_28/mul:z:0$lstm_28/while/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_28/while/lstm_cell_28/add_1µ
$lstm_28/while/lstm_cell_28/Sigmoid_2Sigmoid)lstm_28/while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_28/while/lstm_cell_28/Sigmoid_2§
!lstm_28/while/lstm_cell_28/Relu_1Relu$lstm_28/while/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!lstm_28/while/lstm_cell_28/Relu_1Ù
 lstm_28/while/lstm_cell_28/mul_2Mul(lstm_28/while/lstm_cell_28/Sigmoid_2:y:0/lstm_28/while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_28/while/lstm_cell_28/mul_2
2lstm_28/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_28_while_placeholder_1lstm_28_while_placeholder$lstm_28/while/lstm_cell_28/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_28/while/TensorArrayV2Write/TensorListSetIteml
lstm_28/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_28/while/add/y
lstm_28/while/addAddV2lstm_28_while_placeholderlstm_28/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_28/while/addp
lstm_28/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_28/while/add_1/y
lstm_28/while/add_1AddV2(lstm_28_while_lstm_28_while_loop_counterlstm_28/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_28/while/add_1
lstm_28/while/IdentityIdentitylstm_28/while/add_1:z:0^lstm_28/while/NoOp*
T0*
_output_shapes
: 2
lstm_28/while/Identity¦
lstm_28/while/Identity_1Identity.lstm_28_while_lstm_28_while_maximum_iterations^lstm_28/while/NoOp*
T0*
_output_shapes
: 2
lstm_28/while/Identity_1
lstm_28/while/Identity_2Identitylstm_28/while/add:z:0^lstm_28/while/NoOp*
T0*
_output_shapes
: 2
lstm_28/while/Identity_2º
lstm_28/while/Identity_3IdentityBlstm_28/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_28/while/NoOp*
T0*
_output_shapes
: 2
lstm_28/while/Identity_3®
lstm_28/while/Identity_4Identity$lstm_28/while/lstm_cell_28/mul_2:z:0^lstm_28/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/while/Identity_4®
lstm_28/while/Identity_5Identity$lstm_28/while/lstm_cell_28/add_1:z:0^lstm_28/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/while/Identity_5
lstm_28/while/NoOpNoOp2^lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp1^lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp3^lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_28/while/NoOp"9
lstm_28_while_identitylstm_28/while/Identity:output:0"=
lstm_28_while_identity_1!lstm_28/while/Identity_1:output:0"=
lstm_28_while_identity_2!lstm_28/while/Identity_2:output:0"=
lstm_28_while_identity_3!lstm_28/while/Identity_3:output:0"=
lstm_28_while_identity_4!lstm_28/while/Identity_4:output:0"=
lstm_28_while_identity_5!lstm_28/while/Identity_5:output:0"P
%lstm_28_while_lstm_28_strided_slice_1'lstm_28_while_lstm_28_strided_slice_1_0"z
:lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource<lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource_0"|
;lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource=lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource_0"x
9lstm_28_while_lstm_cell_28_matmul_readvariableop_resource;lstm_28_while_lstm_cell_28_matmul_readvariableop_resource_0"È
alstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensorclstm_28_while_tensorarrayv2read_tensorlistgetitem_lstm_28_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2f
1lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp1lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp2d
0lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp0lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp2h
2lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp2lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 

½
J__inference_sequential_12_layer_call_and_return_conditional_losses_1061925

inputsF
3lstm_26_lstm_cell_26_matmul_readvariableop_resource:	¨I
5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨C
4lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	¨G
3lstm_27_lstm_cell_27_matmul_readvariableop_resource:
ª¨I
5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨C
4lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	¨G
3lstm_28_lstm_cell_28_matmul_readvariableop_resource:
ª¨I
5lstm_28_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨C
4lstm_28_lstm_cell_28_biasadd_readvariableop_resource:	¨>
*dense_29_tensordot_readvariableop_resource:
ªª7
(dense_29_biasadd_readvariableop_resource:	ª=
*dense_30_tensordot_readvariableop_resource:	ª6
(dense_30_biasadd_readvariableop_resource:
identity¢dense_29/BiasAdd/ReadVariableOp¢!dense_29/Tensordot/ReadVariableOp¢dense_30/BiasAdd/ReadVariableOp¢!dense_30/Tensordot/ReadVariableOp¢+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp¢*lstm_26/lstm_cell_26/MatMul/ReadVariableOp¢,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp¢lstm_26/while¢+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp¢*lstm_27/lstm_cell_27/MatMul/ReadVariableOp¢,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp¢lstm_27/while¢+lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp¢*lstm_28/lstm_cell_28/MatMul/ReadVariableOp¢,lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp¢lstm_28/whileT
lstm_26/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_26/Shape
lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice/stack
lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_1
lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_2
lstm_26/strided_sliceStridedSlicelstm_26/Shape:output:0$lstm_26/strided_slice/stack:output:0&lstm_26/strided_slice/stack_1:output:0&lstm_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_26/strided_slices
lstm_26/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_26/zeros/packed/1£
lstm_26/zeros/packedPacklstm_26/strided_slice:output:0lstm_26/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_26/zeros/packedo
lstm_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/zeros/Const
lstm_26/zerosFilllstm_26/zeros/packed:output:0lstm_26/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/zerosw
lstm_26/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_26/zeros_1/packed/1©
lstm_26/zeros_1/packedPacklstm_26/strided_slice:output:0!lstm_26/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_26/zeros_1/packeds
lstm_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/zeros_1/Const
lstm_26/zeros_1Filllstm_26/zeros_1/packed:output:0lstm_26/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/zeros_1
lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose/perm
lstm_26/transpose	Transposeinputslstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_26/transposeg
lstm_26/Shape_1Shapelstm_26/transpose:y:0*
T0*
_output_shapes
:2
lstm_26/Shape_1
lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_1/stack
lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_1
lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_2
lstm_26/strided_slice_1StridedSlicelstm_26/Shape_1:output:0&lstm_26/strided_slice_1/stack:output:0(lstm_26/strided_slice_1/stack_1:output:0(lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_26/strided_slice_1
#lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_26/TensorArrayV2/element_shapeÒ
lstm_26/TensorArrayV2TensorListReserve,lstm_26/TensorArrayV2/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_26/TensorArrayV2Ï
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2?
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_26/transpose:y:0Flstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_26/TensorArrayUnstack/TensorListFromTensor
lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_2/stack
lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_1
lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_2¬
lstm_26/strided_slice_2StridedSlicelstm_26/transpose:y:0&lstm_26/strided_slice_2/stack:output:0(lstm_26/strided_slice_2/stack_1:output:0(lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_26/strided_slice_2Í
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02,
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpÍ
lstm_26/lstm_cell_26/MatMulMatMul lstm_26/strided_slice_2:output:02lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_26/lstm_cell_26/MatMulÔ
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02.
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpÉ
lstm_26/lstm_cell_26/MatMul_1MatMullstm_26/zeros:output:04lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_26/lstm_cell_26/MatMul_1À
lstm_26/lstm_cell_26/addAddV2%lstm_26/lstm_cell_26/MatMul:product:0'lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_26/lstm_cell_26/addÌ
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02-
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpÍ
lstm_26/lstm_cell_26/BiasAddBiasAddlstm_26/lstm_cell_26/add:z:03lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_26/lstm_cell_26/BiasAdd
$lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_26/lstm_cell_26/split/split_dim
lstm_26/lstm_cell_26/splitSplit-lstm_26/lstm_cell_26/split/split_dim:output:0%lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_26/lstm_cell_26/split
lstm_26/lstm_cell_26/SigmoidSigmoid#lstm_26/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/Sigmoid£
lstm_26/lstm_cell_26/Sigmoid_1Sigmoid#lstm_26/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_26/lstm_cell_26/Sigmoid_1¬
lstm_26/lstm_cell_26/mulMul"lstm_26/lstm_cell_26/Sigmoid_1:y:0lstm_26/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/mul
lstm_26/lstm_cell_26/ReluRelu#lstm_26/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/Relu½
lstm_26/lstm_cell_26/mul_1Mul lstm_26/lstm_cell_26/Sigmoid:y:0'lstm_26/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/mul_1²
lstm_26/lstm_cell_26/add_1AddV2lstm_26/lstm_cell_26/mul:z:0lstm_26/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/add_1£
lstm_26/lstm_cell_26/Sigmoid_2Sigmoid#lstm_26/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_26/lstm_cell_26/Sigmoid_2
lstm_26/lstm_cell_26/Relu_1Relulstm_26/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/Relu_1Á
lstm_26/lstm_cell_26/mul_2Mul"lstm_26/lstm_cell_26/Sigmoid_2:y:0)lstm_26/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/lstm_cell_26/mul_2
%lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2'
%lstm_26/TensorArrayV2_1/element_shapeØ
lstm_26/TensorArrayV2_1TensorListReserve.lstm_26/TensorArrayV2_1/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_26/TensorArrayV2_1^
lstm_26/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_26/time
 lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_26/while/maximum_iterationsz
lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_26/while/loop_counter
lstm_26/whileWhile#lstm_26/while/loop_counter:output:0)lstm_26/while/maximum_iterations:output:0lstm_26/time:output:0 lstm_26/TensorArrayV2_1:handle:0lstm_26/zeros:output:0lstm_26/zeros_1:output:0 lstm_26/strided_slice_1:output:0?lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_26_lstm_cell_26_matmul_readvariableop_resource5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_26_while_body_1061478*&
condR
lstm_26_while_cond_1061477*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
lstm_26/whileÅ
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2:
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_26/TensorArrayV2Stack/TensorListStackTensorListStacklstm_26/while:output:3Alstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02,
*lstm_26/TensorArrayV2Stack/TensorListStack
lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_26/strided_slice_3/stack
lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_26/strided_slice_3/stack_1
lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_3/stack_2Ë
lstm_26/strided_slice_3StridedSlice3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_26/strided_slice_3/stack:output:0(lstm_26/strided_slice_3/stack_1:output:0(lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
lstm_26/strided_slice_3
lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose_1/permÆ
lstm_26/transpose_1	Transpose3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_26/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/transpose_1v
lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/runtimey
dropout_43/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_43/dropout/Constª
dropout_43/dropout/MulMullstm_26/transpose_1:y:0!dropout_43/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_43/dropout/Mul{
dropout_43/dropout/ShapeShapelstm_26/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_43/dropout/ShapeÚ
/dropout_43/dropout/random_uniform/RandomUniformRandomUniform!dropout_43/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
dtype021
/dropout_43/dropout/random_uniform/RandomUniform
!dropout_43/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_43/dropout/GreaterEqual/yï
dropout_43/dropout/GreaterEqualGreaterEqual8dropout_43/dropout/random_uniform/RandomUniform:output:0*dropout_43/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
dropout_43/dropout/GreaterEqual¥
dropout_43/dropout/CastCast#dropout_43/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_43/dropout/Cast«
dropout_43/dropout/Mul_1Muldropout_43/dropout/Mul:z:0dropout_43/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_43/dropout/Mul_1j
lstm_27/ShapeShapedropout_43/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_27/Shape
lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice/stack
lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_1
lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_2
lstm_27/strided_sliceStridedSlicelstm_27/Shape:output:0$lstm_27/strided_slice/stack:output:0&lstm_27/strided_slice/stack_1:output:0&lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_27/strided_slices
lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_27/zeros/packed/1£
lstm_27/zeros/packedPacklstm_27/strided_slice:output:0lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_27/zeros/packedo
lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/zeros/Const
lstm_27/zerosFilllstm_27/zeros/packed:output:0lstm_27/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/zerosw
lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_27/zeros_1/packed/1©
lstm_27/zeros_1/packedPacklstm_27/strided_slice:output:0!lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_27/zeros_1/packeds
lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/zeros_1/Const
lstm_27/zeros_1Filllstm_27/zeros_1/packed:output:0lstm_27/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/zeros_1
lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose/perm©
lstm_27/transpose	Transposedropout_43/dropout/Mul_1:z:0lstm_27/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/transposeg
lstm_27/Shape_1Shapelstm_27/transpose:y:0*
T0*
_output_shapes
:2
lstm_27/Shape_1
lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_1/stack
lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_1
lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_2
lstm_27/strided_slice_1StridedSlicelstm_27/Shape_1:output:0&lstm_27/strided_slice_1/stack:output:0(lstm_27/strided_slice_1/stack_1:output:0(lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_27/strided_slice_1
#lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_27/TensorArrayV2/element_shapeÒ
lstm_27/TensorArrayV2TensorListReserve,lstm_27/TensorArrayV2/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_27/TensorArrayV2Ï
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2?
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_27/transpose:y:0Flstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_27/TensorArrayUnstack/TensorListFromTensor
lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_2/stack
lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_1
lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_2­
lstm_27/strided_slice_2StridedSlicelstm_27/transpose:y:0&lstm_27/strided_slice_2/stack:output:0(lstm_27/strided_slice_2/stack_1:output:0(lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
lstm_27/strided_slice_2Î
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3lstm_27_lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02,
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpÍ
lstm_27/lstm_cell_27/MatMulMatMul lstm_27/strided_slice_2:output:02lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_27/lstm_cell_27/MatMulÔ
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02.
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpÉ
lstm_27/lstm_cell_27/MatMul_1MatMullstm_27/zeros:output:04lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_27/lstm_cell_27/MatMul_1À
lstm_27/lstm_cell_27/addAddV2%lstm_27/lstm_cell_27/MatMul:product:0'lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_27/lstm_cell_27/addÌ
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02-
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpÍ
lstm_27/lstm_cell_27/BiasAddBiasAddlstm_27/lstm_cell_27/add:z:03lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_27/lstm_cell_27/BiasAdd
$lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_27/lstm_cell_27/split/split_dim
lstm_27/lstm_cell_27/splitSplit-lstm_27/lstm_cell_27/split/split_dim:output:0%lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_27/lstm_cell_27/split
lstm_27/lstm_cell_27/SigmoidSigmoid#lstm_27/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/Sigmoid£
lstm_27/lstm_cell_27/Sigmoid_1Sigmoid#lstm_27/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_27/lstm_cell_27/Sigmoid_1¬
lstm_27/lstm_cell_27/mulMul"lstm_27/lstm_cell_27/Sigmoid_1:y:0lstm_27/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/mul
lstm_27/lstm_cell_27/ReluRelu#lstm_27/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/Relu½
lstm_27/lstm_cell_27/mul_1Mul lstm_27/lstm_cell_27/Sigmoid:y:0'lstm_27/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/mul_1²
lstm_27/lstm_cell_27/add_1AddV2lstm_27/lstm_cell_27/mul:z:0lstm_27/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/add_1£
lstm_27/lstm_cell_27/Sigmoid_2Sigmoid#lstm_27/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_27/lstm_cell_27/Sigmoid_2
lstm_27/lstm_cell_27/Relu_1Relulstm_27/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/Relu_1Á
lstm_27/lstm_cell_27/mul_2Mul"lstm_27/lstm_cell_27/Sigmoid_2:y:0)lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/lstm_cell_27/mul_2
%lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2'
%lstm_27/TensorArrayV2_1/element_shapeØ
lstm_27/TensorArrayV2_1TensorListReserve.lstm_27/TensorArrayV2_1/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_27/TensorArrayV2_1^
lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_27/time
 lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_27/while/maximum_iterationsz
lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_27/while/loop_counter
lstm_27/whileWhile#lstm_27/while/loop_counter:output:0)lstm_27/while/maximum_iterations:output:0lstm_27/time:output:0 lstm_27/TensorArrayV2_1:handle:0lstm_27/zeros:output:0lstm_27/zeros_1:output:0 lstm_27/strided_slice_1:output:0?lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_27_lstm_cell_27_matmul_readvariableop_resource5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_27_while_body_1061625*&
condR
lstm_27_while_cond_1061624*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
lstm_27/whileÅ
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2:
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_27/TensorArrayV2Stack/TensorListStackTensorListStacklstm_27/while:output:3Alstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02,
*lstm_27/TensorArrayV2Stack/TensorListStack
lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_27/strided_slice_3/stack
lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_27/strided_slice_3/stack_1
lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_3/stack_2Ë
lstm_27/strided_slice_3StridedSlice3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_27/strided_slice_3/stack:output:0(lstm_27/strided_slice_3/stack_1:output:0(lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
lstm_27/strided_slice_3
lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose_1/permÆ
lstm_27/transpose_1	Transpose3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_27/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/transpose_1v
lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/runtimey
dropout_44/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_44/dropout/Constª
dropout_44/dropout/MulMullstm_27/transpose_1:y:0!dropout_44/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_44/dropout/Mul{
dropout_44/dropout/ShapeShapelstm_27/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_44/dropout/ShapeÚ
/dropout_44/dropout/random_uniform/RandomUniformRandomUniform!dropout_44/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
dtype021
/dropout_44/dropout/random_uniform/RandomUniform
!dropout_44/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_44/dropout/GreaterEqual/yï
dropout_44/dropout/GreaterEqualGreaterEqual8dropout_44/dropout/random_uniform/RandomUniform:output:0*dropout_44/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
dropout_44/dropout/GreaterEqual¥
dropout_44/dropout/CastCast#dropout_44/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_44/dropout/Cast«
dropout_44/dropout/Mul_1Muldropout_44/dropout/Mul:z:0dropout_44/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_44/dropout/Mul_1j
lstm_28/ShapeShapedropout_44/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_28/Shape
lstm_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_28/strided_slice/stack
lstm_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_28/strided_slice/stack_1
lstm_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_28/strided_slice/stack_2
lstm_28/strided_sliceStridedSlicelstm_28/Shape:output:0$lstm_28/strided_slice/stack:output:0&lstm_28/strided_slice/stack_1:output:0&lstm_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_28/strided_slices
lstm_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_28/zeros/packed/1£
lstm_28/zeros/packedPacklstm_28/strided_slice:output:0lstm_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_28/zeros/packedo
lstm_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_28/zeros/Const
lstm_28/zerosFilllstm_28/zeros/packed:output:0lstm_28/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/zerosw
lstm_28/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
lstm_28/zeros_1/packed/1©
lstm_28/zeros_1/packedPacklstm_28/strided_slice:output:0!lstm_28/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_28/zeros_1/packeds
lstm_28/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_28/zeros_1/Const
lstm_28/zeros_1Filllstm_28/zeros_1/packed:output:0lstm_28/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/zeros_1
lstm_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_28/transpose/perm©
lstm_28/transpose	Transposedropout_44/dropout/Mul_1:z:0lstm_28/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/transposeg
lstm_28/Shape_1Shapelstm_28/transpose:y:0*
T0*
_output_shapes
:2
lstm_28/Shape_1
lstm_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_28/strided_slice_1/stack
lstm_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_28/strided_slice_1/stack_1
lstm_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_28/strided_slice_1/stack_2
lstm_28/strided_slice_1StridedSlicelstm_28/Shape_1:output:0&lstm_28/strided_slice_1/stack:output:0(lstm_28/strided_slice_1/stack_1:output:0(lstm_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_28/strided_slice_1
#lstm_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_28/TensorArrayV2/element_shapeÒ
lstm_28/TensorArrayV2TensorListReserve,lstm_28/TensorArrayV2/element_shape:output:0 lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_28/TensorArrayV2Ï
=lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2?
=lstm_28/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_28/transpose:y:0Flstm_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_28/TensorArrayUnstack/TensorListFromTensor
lstm_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_28/strided_slice_2/stack
lstm_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_28/strided_slice_2/stack_1
lstm_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_28/strided_slice_2/stack_2­
lstm_28/strided_slice_2StridedSlicelstm_28/transpose:y:0&lstm_28/strided_slice_2/stack:output:0(lstm_28/strided_slice_2/stack_1:output:0(lstm_28/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
lstm_28/strided_slice_2Î
*lstm_28/lstm_cell_28/MatMul/ReadVariableOpReadVariableOp3lstm_28_lstm_cell_28_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02,
*lstm_28/lstm_cell_28/MatMul/ReadVariableOpÍ
lstm_28/lstm_cell_28/MatMulMatMul lstm_28/strided_slice_2:output:02lstm_28/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_28/lstm_cell_28/MatMulÔ
,lstm_28/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp5lstm_28_lstm_cell_28_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02.
,lstm_28/lstm_cell_28/MatMul_1/ReadVariableOpÉ
lstm_28/lstm_cell_28/MatMul_1MatMullstm_28/zeros:output:04lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_28/lstm_cell_28/MatMul_1À
lstm_28/lstm_cell_28/addAddV2%lstm_28/lstm_cell_28/MatMul:product:0'lstm_28/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_28/lstm_cell_28/addÌ
+lstm_28/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp4lstm_28_lstm_cell_28_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02-
+lstm_28/lstm_cell_28/BiasAdd/ReadVariableOpÍ
lstm_28/lstm_cell_28/BiasAddBiasAddlstm_28/lstm_cell_28/add:z:03lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_28/lstm_cell_28/BiasAdd
$lstm_28/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_28/lstm_cell_28/split/split_dim
lstm_28/lstm_cell_28/splitSplit-lstm_28/lstm_cell_28/split/split_dim:output:0%lstm_28/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_28/lstm_cell_28/split
lstm_28/lstm_cell_28/SigmoidSigmoid#lstm_28/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/Sigmoid£
lstm_28/lstm_cell_28/Sigmoid_1Sigmoid#lstm_28/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_28/lstm_cell_28/Sigmoid_1¬
lstm_28/lstm_cell_28/mulMul"lstm_28/lstm_cell_28/Sigmoid_1:y:0lstm_28/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/mul
lstm_28/lstm_cell_28/ReluRelu#lstm_28/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/Relu½
lstm_28/lstm_cell_28/mul_1Mul lstm_28/lstm_cell_28/Sigmoid:y:0'lstm_28/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/mul_1²
lstm_28/lstm_cell_28/add_1AddV2lstm_28/lstm_cell_28/mul:z:0lstm_28/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/add_1£
lstm_28/lstm_cell_28/Sigmoid_2Sigmoid#lstm_28/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_28/lstm_cell_28/Sigmoid_2
lstm_28/lstm_cell_28/Relu_1Relulstm_28/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/Relu_1Á
lstm_28/lstm_cell_28/mul_2Mul"lstm_28/lstm_cell_28/Sigmoid_2:y:0)lstm_28/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/lstm_cell_28/mul_2
%lstm_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2'
%lstm_28/TensorArrayV2_1/element_shapeØ
lstm_28/TensorArrayV2_1TensorListReserve.lstm_28/TensorArrayV2_1/element_shape:output:0 lstm_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_28/TensorArrayV2_1^
lstm_28/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_28/time
 lstm_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_28/while/maximum_iterationsz
lstm_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_28/while/loop_counter
lstm_28/whileWhile#lstm_28/while/loop_counter:output:0)lstm_28/while/maximum_iterations:output:0lstm_28/time:output:0 lstm_28/TensorArrayV2_1:handle:0lstm_28/zeros:output:0lstm_28/zeros_1:output:0 lstm_28/strided_slice_1:output:0?lstm_28/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_28_lstm_cell_28_matmul_readvariableop_resource5lstm_28_lstm_cell_28_matmul_1_readvariableop_resource4lstm_28_lstm_cell_28_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_28_while_body_1061772*&
condR
lstm_28_while_cond_1061771*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
lstm_28/whileÅ
8lstm_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2:
8lstm_28/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_28/TensorArrayV2Stack/TensorListStackTensorListStacklstm_28/while:output:3Alstm_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02,
*lstm_28/TensorArrayV2Stack/TensorListStack
lstm_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_28/strided_slice_3/stack
lstm_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_28/strided_slice_3/stack_1
lstm_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_28/strided_slice_3/stack_2Ë
lstm_28/strided_slice_3StridedSlice3lstm_28/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_28/strided_slice_3/stack:output:0(lstm_28/strided_slice_3/stack_1:output:0(lstm_28/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
lstm_28/strided_slice_3
lstm_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_28/transpose_1/permÆ
lstm_28/transpose_1	Transpose3lstm_28/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_28/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_28/transpose_1v
lstm_28/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_28/runtimey
dropout_45/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_45/dropout/Constª
dropout_45/dropout/MulMullstm_28/transpose_1:y:0!dropout_45/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_45/dropout/Mul{
dropout_45/dropout/ShapeShapelstm_28/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_45/dropout/ShapeÚ
/dropout_45/dropout/random_uniform/RandomUniformRandomUniform!dropout_45/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
dtype021
/dropout_45/dropout/random_uniform/RandomUniform
!dropout_45/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_45/dropout/GreaterEqual/yï
dropout_45/dropout/GreaterEqualGreaterEqual8dropout_45/dropout/random_uniform/RandomUniform:output:0*dropout_45/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
dropout_45/dropout/GreaterEqual¥
dropout_45/dropout/CastCast#dropout_45/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_45/dropout/Cast«
dropout_45/dropout/Mul_1Muldropout_45/dropout/Mul:z:0dropout_45/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_45/dropout/Mul_1³
!dense_29/Tensordot/ReadVariableOpReadVariableOp*dense_29_tensordot_readvariableop_resource* 
_output_shapes
:
ªª*
dtype02#
!dense_29/Tensordot/ReadVariableOp|
dense_29/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_29/Tensordot/axes
dense_29/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_29/Tensordot/free
dense_29/Tensordot/ShapeShapedropout_45/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_29/Tensordot/Shape
 dense_29/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/GatherV2/axisþ
dense_29/Tensordot/GatherV2GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/free:output:0)dense_29/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2
"dense_29/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_29/Tensordot/GatherV2_1/axis
dense_29/Tensordot/GatherV2_1GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/axes:output:0+dense_29/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2_1~
dense_29/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const¤
dense_29/Tensordot/ProdProd$dense_29/Tensordot/GatherV2:output:0!dense_29/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod
dense_29/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const_1¬
dense_29/Tensordot/Prod_1Prod&dense_29/Tensordot/GatherV2_1:output:0#dense_29/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod_1
dense_29/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_29/Tensordot/concat/axisÝ
dense_29/Tensordot/concatConcatV2 dense_29/Tensordot/free:output:0 dense_29/Tensordot/axes:output:0'dense_29/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat°
dense_29/Tensordot/stackPack dense_29/Tensordot/Prod:output:0"dense_29/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/stackÂ
dense_29/Tensordot/transpose	Transposedropout_45/dropout/Mul_1:z:0"dense_29/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_29/Tensordot/transposeÃ
dense_29/Tensordot/ReshapeReshape dense_29/Tensordot/transpose:y:0!dense_29/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_29/Tensordot/ReshapeÃ
dense_29/Tensordot/MatMulMatMul#dense_29/Tensordot/Reshape:output:0)dense_29/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_29/Tensordot/MatMul
dense_29/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ª2
dense_29/Tensordot/Const_2
 dense_29/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/concat_1/axisê
dense_29/Tensordot/concat_1ConcatV2$dense_29/Tensordot/GatherV2:output:0#dense_29/Tensordot/Const_2:output:0)dense_29/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat_1µ
dense_29/TensordotReshape#dense_29/Tensordot/MatMul:product:0$dense_29/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_29/Tensordot¨
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes	
:ª*
dtype02!
dense_29/BiasAdd/ReadVariableOp¬
dense_29/BiasAddBiasAdddense_29/Tensordot:output:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_29/BiasAddx
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_29/Reluy
dropout_46/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_46/dropout/Const®
dropout_46/dropout/MulMuldense_29/Relu:activations:0!dropout_46/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_46/dropout/Mul
dropout_46/dropout/ShapeShapedense_29/Relu:activations:0*
T0*
_output_shapes
:2
dropout_46/dropout/ShapeÚ
/dropout_46/dropout/random_uniform/RandomUniformRandomUniform!dropout_46/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
dtype021
/dropout_46/dropout/random_uniform/RandomUniform
!dropout_46/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2#
!dropout_46/dropout/GreaterEqual/yï
dropout_46/dropout/GreaterEqualGreaterEqual8dropout_46/dropout/random_uniform/RandomUniform:output:0*dropout_46/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
dropout_46/dropout/GreaterEqual¥
dropout_46/dropout/CastCast#dropout_46/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_46/dropout/Cast«
dropout_46/dropout/Mul_1Muldropout_46/dropout/Mul:z:0dropout_46/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout_46/dropout/Mul_1²
!dense_30/Tensordot/ReadVariableOpReadVariableOp*dense_30_tensordot_readvariableop_resource*
_output_shapes
:	ª*
dtype02#
!dense_30/Tensordot/ReadVariableOp|
dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_30/Tensordot/axes
dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_30/Tensordot/free
dense_30/Tensordot/ShapeShapedropout_46/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_30/Tensordot/Shape
 dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/GatherV2/axisþ
dense_30/Tensordot/GatherV2GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/free:output:0)dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2
"dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_30/Tensordot/GatherV2_1/axis
dense_30/Tensordot/GatherV2_1GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/axes:output:0+dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2_1~
dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const¤
dense_30/Tensordot/ProdProd$dense_30/Tensordot/GatherV2:output:0!dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod
dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const_1¬
dense_30/Tensordot/Prod_1Prod&dense_30/Tensordot/GatherV2_1:output:0#dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod_1
dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_30/Tensordot/concat/axisÝ
dense_30/Tensordot/concatConcatV2 dense_30/Tensordot/free:output:0 dense_30/Tensordot/axes:output:0'dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat°
dense_30/Tensordot/stackPack dense_30/Tensordot/Prod:output:0"dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/stackÂ
dense_30/Tensordot/transpose	Transposedropout_46/dropout/Mul_1:z:0"dense_30/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dense_30/Tensordot/transposeÃ
dense_30/Tensordot/ReshapeReshape dense_30/Tensordot/transpose:y:0!dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_30/Tensordot/ReshapeÂ
dense_30/Tensordot/MatMulMatMul#dense_30/Tensordot/Reshape:output:0)dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_30/Tensordot/MatMul
dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_30/Tensordot/Const_2
 dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/concat_1/axisê
dense_30/Tensordot/concat_1ConcatV2$dense_30/Tensordot/GatherV2:output:0#dense_30/Tensordot/Const_2:output:0)dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat_1´
dense_30/TensordotReshape#dense_30/Tensordot/MatMul:product:0$dense_30/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_30/Tensordot§
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_30/BiasAdd/ReadVariableOp«
dense_30/BiasAddBiasAdddense_30/Tensordot:output:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_30/BiasAddx
IdentityIdentitydense_30/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¨
NoOpNoOp ^dense_29/BiasAdd/ReadVariableOp"^dense_29/Tensordot/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp"^dense_30/Tensordot/ReadVariableOp,^lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+^lstm_26/lstm_cell_26/MatMul/ReadVariableOp-^lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^lstm_26/while,^lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+^lstm_27/lstm_cell_27/MatMul/ReadVariableOp-^lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^lstm_27/while,^lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp+^lstm_28/lstm_cell_28/MatMul/ReadVariableOp-^lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp^lstm_28/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2F
!dense_29/Tensordot/ReadVariableOp!dense_29/Tensordot/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2F
!dense_30/Tensordot/ReadVariableOp!dense_30/Tensordot/ReadVariableOp2Z
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp2X
*lstm_26/lstm_cell_26/MatMul/ReadVariableOp*lstm_26/lstm_cell_26/MatMul/ReadVariableOp2\
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp2
lstm_26/whilelstm_26/while2Z
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2X
*lstm_27/lstm_cell_27/MatMul/ReadVariableOp*lstm_27/lstm_cell_27/MatMul/ReadVariableOp2\
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2
lstm_27/whilelstm_27/while2Z
+lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp+lstm_28/lstm_cell_28/BiasAdd/ReadVariableOp2X
*lstm_28/lstm_cell_28/MatMul/ReadVariableOp*lstm_28/lstm_cell_28/MatMul/ReadVariableOp2\
,lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp,lstm_28/lstm_cell_28/MatMul_1/ReadVariableOp2
lstm_28/whilelstm_28/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

æ
/__inference_sequential_12_layer_call_fn_1060023
lstm_26_input
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
	unknown_2:
ª¨
	unknown_3:
ª¨
	unknown_4:	¨
	unknown_5:
ª¨
	unknown_6:
ª¨
	unknown_7:	¨
	unknown_8:
ªª
	unknown_9:	ª

unknown_10:	ª

unknown_11:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
J__inference_sequential_12_layer_call_and_return_conditional_losses_10599942
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
_user_specified_namelstm_26_input
ÃU

D__inference_lstm_28_layer_call_and_return_conditional_losses_1059898

inputs?
+lstm_cell_28_matmul_readvariableop_resource:
ª¨A
-lstm_cell_28_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_28_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_28/BiasAdd/ReadVariableOp¢"lstm_cell_28/MatMul/ReadVariableOp¢$lstm_cell_28/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
:ÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_28/MatMul/ReadVariableOpReadVariableOp+lstm_cell_28_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_28/MatMul/ReadVariableOp­
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0*lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul¼
$lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_28_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_28/MatMul_1/ReadVariableOp©
lstm_cell_28/MatMul_1MatMulzeros:output:0,lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul_1 
lstm_cell_28/addAddV2lstm_cell_28/MatMul:product:0lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/add´
#lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_28_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_28/BiasAdd/ReadVariableOp­
lstm_cell_28/BiasAddBiasAddlstm_cell_28/add:z:0+lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/BiasAdd~
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_28/split/split_dim÷
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_28/split
lstm_cell_28/SigmoidSigmoidlstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid
lstm_cell_28/Sigmoid_1Sigmoidlstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_1
lstm_cell_28/mulMullstm_cell_28/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul~
lstm_cell_28/ReluRelulstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu
lstm_cell_28/mul_1Mullstm_cell_28/Sigmoid:y:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_1
lstm_cell_28/add_1AddV2lstm_cell_28/mul:z:0lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/add_1
lstm_cell_28/Sigmoid_2Sigmoidlstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_2}
lstm_cell_28/Relu_1Relulstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu_1¡
lstm_cell_28/mul_2Mullstm_cell_28/Sigmoid_2:y:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_28_matmul_readvariableop_resource-lstm_cell_28_matmul_1_readvariableop_resource,lstm_cell_28_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1059814*
condR
while_cond_1059813*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_28/BiasAdd/ReadVariableOp#^lstm_cell_28/MatMul/ReadVariableOp%^lstm_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_28/BiasAdd/ReadVariableOp#lstm_cell_28/BiasAdd/ReadVariableOp2H
"lstm_cell_28/MatMul/ReadVariableOp"lstm_cell_28/MatMul/ReadVariableOp2L
$lstm_cell_28/MatMul_1/ReadVariableOp$lstm_cell_28/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs

e
G__inference_dropout_46_layer_call_and_return_conditional_losses_1059955

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
¯?
Ó
while_body_1062457
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	¨I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	¨G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_26_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	¨*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_26/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Ö
f
G__inference_dropout_44_layer_call_and_return_conditional_losses_1060274

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
:ÿÿÿÿÿÿÿÿÿª2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
³?
Õ
while_body_1063100
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_27_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
ç^

(sequential_12_lstm_27_while_body_1057362H
Dsequential_12_lstm_27_while_sequential_12_lstm_27_while_loop_counterN
Jsequential_12_lstm_27_while_sequential_12_lstm_27_while_maximum_iterations+
'sequential_12_lstm_27_while_placeholder-
)sequential_12_lstm_27_while_placeholder_1-
)sequential_12_lstm_27_while_placeholder_2-
)sequential_12_lstm_27_while_placeholder_3G
Csequential_12_lstm_27_while_sequential_12_lstm_27_strided_slice_1_0
sequential_12_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_27_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_12_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:
ª¨_
Ksequential_12_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ª¨Y
Jsequential_12_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	¨(
$sequential_12_lstm_27_while_identity*
&sequential_12_lstm_27_while_identity_1*
&sequential_12_lstm_27_while_identity_2*
&sequential_12_lstm_27_while_identity_3*
&sequential_12_lstm_27_while_identity_4*
&sequential_12_lstm_27_while_identity_5E
Asequential_12_lstm_27_while_sequential_12_lstm_27_strided_slice_1
}sequential_12_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_27_tensorarrayunstack_tensorlistfromtensor[
Gsequential_12_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:
ª¨]
Isequential_12_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨W
Hsequential_12_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	¨¢?sequential_12/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp¢>sequential_12/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp¢@sequential_12/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpï
Msequential_12/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2O
Msequential_12/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeØ
?sequential_12/lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_27_tensorarrayunstack_tensorlistfromtensor_0'sequential_12_lstm_27_while_placeholderVsequential_12/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02A
?sequential_12/lstm_27/while/TensorArrayV2Read/TensorListGetItem
>sequential_12/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOpIsequential_12_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02@
>sequential_12/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp¯
/sequential_12/lstm_27/while/lstm_cell_27/MatMulMatMulFsequential_12/lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨21
/sequential_12/lstm_27/while/lstm_cell_27/MatMul
@sequential_12/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02B
@sequential_12/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp
1sequential_12/lstm_27/while/lstm_cell_27/MatMul_1MatMul)sequential_12_lstm_27_while_placeholder_2Hsequential_12/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨23
1sequential_12/lstm_27/while/lstm_cell_27/MatMul_1
,sequential_12/lstm_27/while/lstm_cell_27/addAddV29sequential_12/lstm_27/while/lstm_cell_27/MatMul:product:0;sequential_12/lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2.
,sequential_12/lstm_27/while/lstm_cell_27/add
?sequential_12/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02A
?sequential_12/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp
0sequential_12/lstm_27/while/lstm_cell_27/BiasAddBiasAdd0sequential_12/lstm_27/while/lstm_cell_27/add:z:0Gsequential_12/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨22
0sequential_12/lstm_27/while/lstm_cell_27/BiasAdd¶
8sequential_12/lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_27/while/lstm_cell_27/split/split_dimç
.sequential_12/lstm_27/while/lstm_cell_27/splitSplitAsequential_12/lstm_27/while/lstm_cell_27/split/split_dim:output:09sequential_12/lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split20
.sequential_12/lstm_27/while/lstm_cell_27/splitÛ
0sequential_12/lstm_27/while/lstm_cell_27/SigmoidSigmoid7sequential_12/lstm_27/while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª22
0sequential_12/lstm_27/while/lstm_cell_27/Sigmoidß
2sequential_12/lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid7sequential_12/lstm_27/while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª24
2sequential_12/lstm_27/while/lstm_cell_27/Sigmoid_1ù
,sequential_12/lstm_27/while/lstm_cell_27/mulMul6sequential_12/lstm_27/while/lstm_cell_27/Sigmoid_1:y:0)sequential_12_lstm_27_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2.
,sequential_12/lstm_27/while/lstm_cell_27/mulÒ
-sequential_12/lstm_27/while/lstm_cell_27/ReluRelu7sequential_12/lstm_27/while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2/
-sequential_12/lstm_27/while/lstm_cell_27/Relu
.sequential_12/lstm_27/while/lstm_cell_27/mul_1Mul4sequential_12/lstm_27/while/lstm_cell_27/Sigmoid:y:0;sequential_12/lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª20
.sequential_12/lstm_27/while/lstm_cell_27/mul_1
.sequential_12/lstm_27/while/lstm_cell_27/add_1AddV20sequential_12/lstm_27/while/lstm_cell_27/mul:z:02sequential_12/lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª20
.sequential_12/lstm_27/while/lstm_cell_27/add_1ß
2sequential_12/lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid7sequential_12/lstm_27/while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª24
2sequential_12/lstm_27/while/lstm_cell_27/Sigmoid_2Ñ
/sequential_12/lstm_27/while/lstm_cell_27/Relu_1Relu2sequential_12/lstm_27/while/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª21
/sequential_12/lstm_27/while/lstm_cell_27/Relu_1
.sequential_12/lstm_27/while/lstm_cell_27/mul_2Mul6sequential_12/lstm_27/while/lstm_cell_27/Sigmoid_2:y:0=sequential_12/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª20
.sequential_12/lstm_27/while/lstm_cell_27/mul_2Î
@sequential_12/lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_12_lstm_27_while_placeholder_1'sequential_12_lstm_27_while_placeholder2sequential_12/lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_12/lstm_27/while/TensorArrayV2Write/TensorListSetItem
!sequential_12/lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_12/lstm_27/while/add/yÁ
sequential_12/lstm_27/while/addAddV2'sequential_12_lstm_27_while_placeholder*sequential_12/lstm_27/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_27/while/add
#sequential_12/lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_12/lstm_27/while/add_1/yä
!sequential_12/lstm_27/while/add_1AddV2Dsequential_12_lstm_27_while_sequential_12_lstm_27_while_loop_counter,sequential_12/lstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_27/while/add_1Ã
$sequential_12/lstm_27/while/IdentityIdentity%sequential_12/lstm_27/while/add_1:z:0!^sequential_12/lstm_27/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_12/lstm_27/while/Identityì
&sequential_12/lstm_27/while/Identity_1IdentityJsequential_12_lstm_27_while_sequential_12_lstm_27_while_maximum_iterations!^sequential_12/lstm_27/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_27/while/Identity_1Å
&sequential_12/lstm_27/while/Identity_2Identity#sequential_12/lstm_27/while/add:z:0!^sequential_12/lstm_27/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_27/while/Identity_2ò
&sequential_12/lstm_27/while/Identity_3IdentityPsequential_12/lstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_12/lstm_27/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_27/while/Identity_3æ
&sequential_12/lstm_27/while/Identity_4Identity2sequential_12/lstm_27/while/lstm_cell_27/mul_2:z:0!^sequential_12/lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2(
&sequential_12/lstm_27/while/Identity_4æ
&sequential_12/lstm_27/while/Identity_5Identity2sequential_12/lstm_27/while/lstm_cell_27/add_1:z:0!^sequential_12/lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2(
&sequential_12/lstm_27/while/Identity_5Ì
 sequential_12/lstm_27/while/NoOpNoOp@^sequential_12/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp?^sequential_12/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpA^sequential_12/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_12/lstm_27/while/NoOp"U
$sequential_12_lstm_27_while_identity-sequential_12/lstm_27/while/Identity:output:0"Y
&sequential_12_lstm_27_while_identity_1/sequential_12/lstm_27/while/Identity_1:output:0"Y
&sequential_12_lstm_27_while_identity_2/sequential_12/lstm_27/while/Identity_2:output:0"Y
&sequential_12_lstm_27_while_identity_3/sequential_12/lstm_27/while/Identity_3:output:0"Y
&sequential_12_lstm_27_while_identity_4/sequential_12/lstm_27/while/Identity_4:output:0"Y
&sequential_12_lstm_27_while_identity_5/sequential_12/lstm_27/while/Identity_5:output:0"
Hsequential_12_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resourceJsequential_12_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"
Isequential_12_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resourceKsequential_12_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"
Gsequential_12_lstm_27_while_lstm_cell_27_matmul_readvariableop_resourceIsequential_12_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"
Asequential_12_lstm_27_while_sequential_12_lstm_27_strided_slice_1Csequential_12_lstm_27_while_sequential_12_lstm_27_strided_slice_1_0"
}sequential_12_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_27_tensorarrayunstack_tensorlistfromtensorsequential_12_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2
?sequential_12/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp?sequential_12/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2
>sequential_12/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp>sequential_12/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2
@sequential_12/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp@sequential_12/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
äJ
Ó

lstm_26_while_body_1061478,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3+
'lstm_26_while_lstm_26_strided_slice_1_0g
clstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	¨Q
=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ª¨K
<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	¨
lstm_26_while_identity
lstm_26_while_identity_1
lstm_26_while_identity_2
lstm_26_while_identity_3
lstm_26_while_identity_4
lstm_26_while_identity_5)
%lstm_26_while_lstm_26_strided_slice_1e
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorL
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	¨O
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:
ª¨I
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	¨¢1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp¢0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp¢2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpÓ
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2A
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0lstm_26_while_placeholderHlstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_26/while/TensorArrayV2Read/TensorListGetItemá
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	¨*
dtype022
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp÷
!lstm_26/while/lstm_cell_26/MatMulMatMul8lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2#
!lstm_26/while/lstm_cell_26/MatMulè
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype024
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpà
#lstm_26/while/lstm_cell_26/MatMul_1MatMullstm_26_while_placeholder_2:lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2%
#lstm_26/while/lstm_cell_26/MatMul_1Ø
lstm_26/while/lstm_cell_26/addAddV2+lstm_26/while/lstm_cell_26/MatMul:product:0-lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2 
lstm_26/while/lstm_cell_26/addà
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype023
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpå
"lstm_26/while/lstm_cell_26/BiasAddBiasAdd"lstm_26/while/lstm_cell_26/add:z:09lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2$
"lstm_26/while/lstm_cell_26/BiasAdd
*lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_26/while/lstm_cell_26/split/split_dim¯
 lstm_26/while/lstm_cell_26/splitSplit3lstm_26/while/lstm_cell_26/split/split_dim:output:0+lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2"
 lstm_26/while/lstm_cell_26/split±
"lstm_26/while/lstm_cell_26/SigmoidSigmoid)lstm_26/while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2$
"lstm_26/while/lstm_cell_26/Sigmoidµ
$lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid)lstm_26/while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_26/while/lstm_cell_26/Sigmoid_1Á
lstm_26/while/lstm_cell_26/mulMul(lstm_26/while/lstm_cell_26/Sigmoid_1:y:0lstm_26_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_26/while/lstm_cell_26/mul¨
lstm_26/while/lstm_cell_26/ReluRelu)lstm_26/while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
lstm_26/while/lstm_cell_26/ReluÕ
 lstm_26/while/lstm_cell_26/mul_1Mul&lstm_26/while/lstm_cell_26/Sigmoid:y:0-lstm_26/while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_26/while/lstm_cell_26/mul_1Ê
 lstm_26/while/lstm_cell_26/add_1AddV2"lstm_26/while/lstm_cell_26/mul:z:0$lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_26/while/lstm_cell_26/add_1µ
$lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid)lstm_26/while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_26/while/lstm_cell_26/Sigmoid_2§
!lstm_26/while/lstm_cell_26/Relu_1Relu$lstm_26/while/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!lstm_26/while/lstm_cell_26/Relu_1Ù
 lstm_26/while/lstm_cell_26/mul_2Mul(lstm_26/while/lstm_cell_26/Sigmoid_2:y:0/lstm_26/while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_26/while/lstm_cell_26/mul_2
2lstm_26/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_26_while_placeholder_1lstm_26_while_placeholder$lstm_26/while/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_26/while/TensorArrayV2Write/TensorListSetIteml
lstm_26/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_26/while/add/y
lstm_26/while/addAddV2lstm_26_while_placeholderlstm_26/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_26/while/addp
lstm_26/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_26/while/add_1/y
lstm_26/while/add_1AddV2(lstm_26_while_lstm_26_while_loop_counterlstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_26/while/add_1
lstm_26/while/IdentityIdentitylstm_26/while/add_1:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity¦
lstm_26/while/Identity_1Identity.lstm_26_while_lstm_26_while_maximum_iterations^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_1
lstm_26/while/Identity_2Identitylstm_26/while/add:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_2º
lstm_26/while/Identity_3IdentityBlstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_3®
lstm_26/while/Identity_4Identity$lstm_26/while/lstm_cell_26/mul_2:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/while/Identity_4®
lstm_26/while/Identity_5Identity$lstm_26/while/lstm_cell_26/add_1:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_26/while/Identity_5
lstm_26/while/NoOpNoOp2^lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1^lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp3^lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_26/while/NoOp"9
lstm_26_while_identitylstm_26/while/Identity:output:0"=
lstm_26_while_identity_1!lstm_26/while/Identity_1:output:0"=
lstm_26_while_identity_2!lstm_26/while/Identity_2:output:0"=
lstm_26_while_identity_3!lstm_26/while/Identity_3:output:0"=
lstm_26_while_identity_4!lstm_26/while/Identity_4:output:0"=
lstm_26_while_identity_5!lstm_26/while/Identity_5:output:0"P
%lstm_26_while_lstm_26_strided_slice_1'lstm_26_while_lstm_26_strided_slice_1_0"z
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0"|
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0"x
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"È
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2f
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp2d
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp2h
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1060536
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1060536___redundant_placeholder05
1while_while_cond_1060536___redundant_placeholder15
1while_while_cond_1060536___redundant_placeholder25
1while_while_cond_1060536___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
³?
Õ
while_body_1059657
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_27_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1059119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1059119___redundant_placeholder05
1while_while_cond_1059119___redundant_placeholder15
1while_while_cond_1059119___redundant_placeholder25
1while_while_cond_1059119___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
¹
e
,__inference_dropout_45_layer_call_fn_1063837

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
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_10600862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
ÃU

D__inference_lstm_28_layer_call_and_return_conditional_losses_1063827

inputs?
+lstm_cell_28_matmul_readvariableop_resource:
ª¨A
-lstm_cell_28_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_28_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_28/BiasAdd/ReadVariableOp¢"lstm_cell_28/MatMul/ReadVariableOp¢$lstm_cell_28/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
:ÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_28/MatMul/ReadVariableOpReadVariableOp+lstm_cell_28_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_28/MatMul/ReadVariableOp­
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0*lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul¼
$lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_28_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_28/MatMul_1/ReadVariableOp©
lstm_cell_28/MatMul_1MatMulzeros:output:0,lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul_1 
lstm_cell_28/addAddV2lstm_cell_28/MatMul:product:0lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/add´
#lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_28_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_28/BiasAdd/ReadVariableOp­
lstm_cell_28/BiasAddBiasAddlstm_cell_28/add:z:0+lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/BiasAdd~
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_28/split/split_dim÷
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_28/split
lstm_cell_28/SigmoidSigmoidlstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid
lstm_cell_28/Sigmoid_1Sigmoidlstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_1
lstm_cell_28/mulMullstm_cell_28/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul~
lstm_cell_28/ReluRelulstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu
lstm_cell_28/mul_1Mullstm_cell_28/Sigmoid:y:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_1
lstm_cell_28/add_1AddV2lstm_cell_28/mul:z:0lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/add_1
lstm_cell_28/Sigmoid_2Sigmoidlstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_2}
lstm_cell_28/Relu_1Relulstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu_1¡
lstm_cell_28/mul_2Mullstm_cell_28/Sigmoid_2:y:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_28_matmul_readvariableop_resource-lstm_cell_28_matmul_1_readvariableop_resource,lstm_cell_28_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1063743*
condR
while_cond_1063742*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_28/BiasAdd/ReadVariableOp#^lstm_cell_28/MatMul/ReadVariableOp%^lstm_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_28/BiasAdd/ReadVariableOp#lstm_cell_28/BiasAdd/ReadVariableOp2H
"lstm_cell_28/MatMul/ReadVariableOp"lstm_cell_28/MatMul/ReadVariableOp2L
$lstm_cell_28/MatMul_1/ReadVariableOp$lstm_cell_28/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Þ
¹
)__inference_lstm_26_layer_call_fn_1061947
inputs_0
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_26_layer_call_and_return_conditional_losses_10579932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

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
½U

D__inference_lstm_26_layer_call_and_return_conditional_losses_1060621

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	¨A
-lstm_cell_26_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_26_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1060537*
condR
while_cond_1060536*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
È
while_cond_1057923
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1057923___redundant_placeholder05
1while_while_cond_1057923___redundant_placeholder15
1while_while_cond_1057923___redundant_placeholder25
1while_while_cond_1057923___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
Ö
f
G__inference_dropout_43_layer_call_and_return_conditional_losses_1060462

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
:ÿÿÿÿÿÿÿÿÿª2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
½U

D__inference_lstm_26_layer_call_and_return_conditional_losses_1062398

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	¨A
-lstm_cell_26_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_26_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1062314*
condR
while_cond_1062313*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
f
G__inference_dropout_46_layer_call_and_return_conditional_losses_1060053

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
:ÿÿÿÿÿÿÿÿÿª2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
´
·
)__inference_lstm_26_layer_call_fn_1061969

inputs
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_26_layer_call_and_return_conditional_losses_10606212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

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
ì 
ý
E__inference_dense_30_layer_call_and_return_conditional_losses_1059987

inputs4
!tensordot_readvariableop_resource:	ª-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	ª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
ô

(sequential_12_lstm_27_while_cond_1057361H
Dsequential_12_lstm_27_while_sequential_12_lstm_27_while_loop_counterN
Jsequential_12_lstm_27_while_sequential_12_lstm_27_while_maximum_iterations+
'sequential_12_lstm_27_while_placeholder-
)sequential_12_lstm_27_while_placeholder_1-
)sequential_12_lstm_27_while_placeholder_2-
)sequential_12_lstm_27_while_placeholder_3J
Fsequential_12_lstm_27_while_less_sequential_12_lstm_27_strided_slice_1a
]sequential_12_lstm_27_while_sequential_12_lstm_27_while_cond_1057361___redundant_placeholder0a
]sequential_12_lstm_27_while_sequential_12_lstm_27_while_cond_1057361___redundant_placeholder1a
]sequential_12_lstm_27_while_sequential_12_lstm_27_while_cond_1057361___redundant_placeholder2a
]sequential_12_lstm_27_while_sequential_12_lstm_27_while_cond_1057361___redundant_placeholder3(
$sequential_12_lstm_27_while_identity
Þ
 sequential_12/lstm_27/while/LessLess'sequential_12_lstm_27_while_placeholderFsequential_12_lstm_27_while_less_sequential_12_lstm_27_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_12/lstm_27/while/Less
$sequential_12/lstm_27/while/IdentityIdentity$sequential_12/lstm_27/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_12/lstm_27/while/Identity"U
$sequential_12_lstm_27_while_identity-sequential_12/lstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
ç^

(sequential_12_lstm_28_while_body_1057502H
Dsequential_12_lstm_28_while_sequential_12_lstm_28_while_loop_counterN
Jsequential_12_lstm_28_while_sequential_12_lstm_28_while_maximum_iterations+
'sequential_12_lstm_28_while_placeholder-
)sequential_12_lstm_28_while_placeholder_1-
)sequential_12_lstm_28_while_placeholder_2-
)sequential_12_lstm_28_while_placeholder_3G
Csequential_12_lstm_28_while_sequential_12_lstm_28_strided_slice_1_0
sequential_12_lstm_28_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_28_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_12_lstm_28_while_lstm_cell_28_matmul_readvariableop_resource_0:
ª¨_
Ksequential_12_lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource_0:
ª¨Y
Jsequential_12_lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource_0:	¨(
$sequential_12_lstm_28_while_identity*
&sequential_12_lstm_28_while_identity_1*
&sequential_12_lstm_28_while_identity_2*
&sequential_12_lstm_28_while_identity_3*
&sequential_12_lstm_28_while_identity_4*
&sequential_12_lstm_28_while_identity_5E
Asequential_12_lstm_28_while_sequential_12_lstm_28_strided_slice_1
}sequential_12_lstm_28_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_28_tensorarrayunstack_tensorlistfromtensor[
Gsequential_12_lstm_28_while_lstm_cell_28_matmul_readvariableop_resource:
ª¨]
Isequential_12_lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨W
Hsequential_12_lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource:	¨¢?sequential_12/lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp¢>sequential_12/lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp¢@sequential_12/lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOpï
Msequential_12/lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2O
Msequential_12/lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shapeØ
?sequential_12/lstm_28/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_lstm_28_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_28_tensorarrayunstack_tensorlistfromtensor_0'sequential_12_lstm_28_while_placeholderVsequential_12/lstm_28/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02A
?sequential_12/lstm_28/while/TensorArrayV2Read/TensorListGetItem
>sequential_12/lstm_28/while/lstm_cell_28/MatMul/ReadVariableOpReadVariableOpIsequential_12_lstm_28_while_lstm_cell_28_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02@
>sequential_12/lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp¯
/sequential_12/lstm_28/while/lstm_cell_28/MatMulMatMulFsequential_12/lstm_28/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨21
/sequential_12/lstm_28/while/lstm_cell_28/MatMul
@sequential_12/lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02B
@sequential_12/lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp
1sequential_12/lstm_28/while/lstm_cell_28/MatMul_1MatMul)sequential_12_lstm_28_while_placeholder_2Hsequential_12/lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨23
1sequential_12/lstm_28/while/lstm_cell_28/MatMul_1
,sequential_12/lstm_28/while/lstm_cell_28/addAddV29sequential_12/lstm_28/while/lstm_cell_28/MatMul:product:0;sequential_12/lstm_28/while/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2.
,sequential_12/lstm_28/while/lstm_cell_28/add
?sequential_12/lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02A
?sequential_12/lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp
0sequential_12/lstm_28/while/lstm_cell_28/BiasAddBiasAdd0sequential_12/lstm_28/while/lstm_cell_28/add:z:0Gsequential_12/lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨22
0sequential_12/lstm_28/while/lstm_cell_28/BiasAdd¶
8sequential_12/lstm_28/while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_28/while/lstm_cell_28/split/split_dimç
.sequential_12/lstm_28/while/lstm_cell_28/splitSplitAsequential_12/lstm_28/while/lstm_cell_28/split/split_dim:output:09sequential_12/lstm_28/while/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split20
.sequential_12/lstm_28/while/lstm_cell_28/splitÛ
0sequential_12/lstm_28/while/lstm_cell_28/SigmoidSigmoid7sequential_12/lstm_28/while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª22
0sequential_12/lstm_28/while/lstm_cell_28/Sigmoidß
2sequential_12/lstm_28/while/lstm_cell_28/Sigmoid_1Sigmoid7sequential_12/lstm_28/while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª24
2sequential_12/lstm_28/while/lstm_cell_28/Sigmoid_1ù
,sequential_12/lstm_28/while/lstm_cell_28/mulMul6sequential_12/lstm_28/while/lstm_cell_28/Sigmoid_1:y:0)sequential_12_lstm_28_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2.
,sequential_12/lstm_28/while/lstm_cell_28/mulÒ
-sequential_12/lstm_28/while/lstm_cell_28/ReluRelu7sequential_12/lstm_28/while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2/
-sequential_12/lstm_28/while/lstm_cell_28/Relu
.sequential_12/lstm_28/while/lstm_cell_28/mul_1Mul4sequential_12/lstm_28/while/lstm_cell_28/Sigmoid:y:0;sequential_12/lstm_28/while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª20
.sequential_12/lstm_28/while/lstm_cell_28/mul_1
.sequential_12/lstm_28/while/lstm_cell_28/add_1AddV20sequential_12/lstm_28/while/lstm_cell_28/mul:z:02sequential_12/lstm_28/while/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª20
.sequential_12/lstm_28/while/lstm_cell_28/add_1ß
2sequential_12/lstm_28/while/lstm_cell_28/Sigmoid_2Sigmoid7sequential_12/lstm_28/while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª24
2sequential_12/lstm_28/while/lstm_cell_28/Sigmoid_2Ñ
/sequential_12/lstm_28/while/lstm_cell_28/Relu_1Relu2sequential_12/lstm_28/while/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª21
/sequential_12/lstm_28/while/lstm_cell_28/Relu_1
.sequential_12/lstm_28/while/lstm_cell_28/mul_2Mul6sequential_12/lstm_28/while/lstm_cell_28/Sigmoid_2:y:0=sequential_12/lstm_28/while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª20
.sequential_12/lstm_28/while/lstm_cell_28/mul_2Î
@sequential_12/lstm_28/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_12_lstm_28_while_placeholder_1'sequential_12_lstm_28_while_placeholder2sequential_12/lstm_28/while/lstm_cell_28/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_12/lstm_28/while/TensorArrayV2Write/TensorListSetItem
!sequential_12/lstm_28/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_12/lstm_28/while/add/yÁ
sequential_12/lstm_28/while/addAddV2'sequential_12_lstm_28_while_placeholder*sequential_12/lstm_28/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_28/while/add
#sequential_12/lstm_28/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_12/lstm_28/while/add_1/yä
!sequential_12/lstm_28/while/add_1AddV2Dsequential_12_lstm_28_while_sequential_12_lstm_28_while_loop_counter,sequential_12/lstm_28/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_28/while/add_1Ã
$sequential_12/lstm_28/while/IdentityIdentity%sequential_12/lstm_28/while/add_1:z:0!^sequential_12/lstm_28/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_12/lstm_28/while/Identityì
&sequential_12/lstm_28/while/Identity_1IdentityJsequential_12_lstm_28_while_sequential_12_lstm_28_while_maximum_iterations!^sequential_12/lstm_28/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_28/while/Identity_1Å
&sequential_12/lstm_28/while/Identity_2Identity#sequential_12/lstm_28/while/add:z:0!^sequential_12/lstm_28/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_28/while/Identity_2ò
&sequential_12/lstm_28/while/Identity_3IdentityPsequential_12/lstm_28/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_12/lstm_28/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_28/while/Identity_3æ
&sequential_12/lstm_28/while/Identity_4Identity2sequential_12/lstm_28/while/lstm_cell_28/mul_2:z:0!^sequential_12/lstm_28/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2(
&sequential_12/lstm_28/while/Identity_4æ
&sequential_12/lstm_28/while/Identity_5Identity2sequential_12/lstm_28/while/lstm_cell_28/add_1:z:0!^sequential_12/lstm_28/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2(
&sequential_12/lstm_28/while/Identity_5Ì
 sequential_12/lstm_28/while/NoOpNoOp@^sequential_12/lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp?^sequential_12/lstm_28/while/lstm_cell_28/MatMul/ReadVariableOpA^sequential_12/lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_12/lstm_28/while/NoOp"U
$sequential_12_lstm_28_while_identity-sequential_12/lstm_28/while/Identity:output:0"Y
&sequential_12_lstm_28_while_identity_1/sequential_12/lstm_28/while/Identity_1:output:0"Y
&sequential_12_lstm_28_while_identity_2/sequential_12/lstm_28/while/Identity_2:output:0"Y
&sequential_12_lstm_28_while_identity_3/sequential_12/lstm_28/while/Identity_3:output:0"Y
&sequential_12_lstm_28_while_identity_4/sequential_12/lstm_28/while/Identity_4:output:0"Y
&sequential_12_lstm_28_while_identity_5/sequential_12/lstm_28/while/Identity_5:output:0"
Hsequential_12_lstm_28_while_lstm_cell_28_biasadd_readvariableop_resourceJsequential_12_lstm_28_while_lstm_cell_28_biasadd_readvariableop_resource_0"
Isequential_12_lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resourceKsequential_12_lstm_28_while_lstm_cell_28_matmul_1_readvariableop_resource_0"
Gsequential_12_lstm_28_while_lstm_cell_28_matmul_readvariableop_resourceIsequential_12_lstm_28_while_lstm_cell_28_matmul_readvariableop_resource_0"
Asequential_12_lstm_28_while_sequential_12_lstm_28_strided_slice_1Csequential_12_lstm_28_while_sequential_12_lstm_28_strided_slice_1_0"
}sequential_12_lstm_28_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_28_tensorarrayunstack_tensorlistfromtensorsequential_12_lstm_28_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_28_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2
?sequential_12/lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp?sequential_12/lstm_28/while/lstm_cell_28/BiasAdd/ReadVariableOp2
>sequential_12/lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp>sequential_12/lstm_28/while/lstm_cell_28/MatMul/ReadVariableOp2
@sequential_12/lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp@sequential_12/lstm_28/while/lstm_cell_28/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
+
å
J__inference_sequential_12_layer_call_and_return_conditional_losses_1060801
lstm_26_input"
lstm_26_1060765:	¨#
lstm_26_1060767:
ª¨
lstm_26_1060769:	¨#
lstm_27_1060773:
ª¨#
lstm_27_1060775:
ª¨
lstm_27_1060777:	¨#
lstm_28_1060781:
ª¨#
lstm_28_1060783:
ª¨
lstm_28_1060785:	¨$
dense_29_1060789:
ªª
dense_29_1060791:	ª#
dense_30_1060795:	ª
dense_30_1060797:
identity¢ dense_29/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢lstm_26/StatefulPartitionedCall¢lstm_27/StatefulPartitionedCall¢lstm_28/StatefulPartitionedCall±
lstm_26/StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputlstm_26_1060765lstm_26_1060767lstm_26_1060769*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_26_layer_call_and_return_conditional_losses_10595842!
lstm_26/StatefulPartitionedCall
dropout_43/PartitionedCallPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_10595972
dropout_43/PartitionedCallÇ
lstm_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_43/PartitionedCall:output:0lstm_27_1060773lstm_27_1060775lstm_27_1060777*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_27_layer_call_and_return_conditional_losses_10597412!
lstm_27/StatefulPartitionedCall
dropout_44/PartitionedCallPartitionedCall(lstm_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_44_layer_call_and_return_conditional_losses_10597542
dropout_44/PartitionedCallÇ
lstm_28/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0lstm_28_1060781lstm_28_1060783lstm_28_1060785*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_28_layer_call_and_return_conditional_losses_10598982!
lstm_28/StatefulPartitionedCall
dropout_45/PartitionedCallPartitionedCall(lstm_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_10599112
dropout_45/PartitionedCall¹
 dense_29/StatefulPartitionedCallStatefulPartitionedCall#dropout_45/PartitionedCall:output:0dense_29_1060789dense_29_1060791*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_10599442"
 dense_29/StatefulPartitionedCall
dropout_46/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_46_layer_call_and_return_conditional_losses_10599552
dropout_46/PartitionedCall¸
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_46/PartitionedCall:output:0dense_30_1060795dense_30_1060797*
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
E__inference_dense_30_layer_call_and_return_conditional_losses_10599872"
 dense_30/StatefulPartitionedCall
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityú
NoOpNoOp!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall ^lstm_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall2B
lstm_28/StatefulPartitionedCalllstm_28/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_26_input

æ
/__inference_sequential_12_layer_call_fn_1060762
lstm_26_input
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
	unknown_2:
ª¨
	unknown_3:
ª¨
	unknown_4:	¨
	unknown_5:
ª¨
	unknown_6:
ª¨
	unknown_7:	¨
	unknown_8:
ªª
	unknown_9:	ª

unknown_10:	ª

unknown_11:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
J__inference_sequential_12_layer_call_and_return_conditional_losses_10607022
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
_user_specified_namelstm_26_input
³?
Õ
while_body_1062957
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_27_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
üU

D__inference_lstm_26_layer_call_and_return_conditional_losses_1062112
inputs_0>
+lstm_cell_26_matmul_readvariableop_resource:	¨A
-lstm_cell_26_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_26_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileF
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	¨*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1062028*
condR
while_cond_1062027*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ú
Ü
%__inference_signature_wrapper_1060879
lstm_26_input
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
	unknown_2:
ª¨
	unknown_3:
ª¨
	unknown_4:	¨
	unknown_5:
ª¨
	unknown_6:
ª¨
	unknown_7:	¨
	unknown_8:
ªª
	unknown_9:	ª

unknown_10:	ª

unknown_11:
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_10576412
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
_user_specified_namelstm_26_input
Þ
È
while_cond_1058917
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1058917___redundant_placeholder05
1while_while_cond_1058917___redundant_placeholder15
1while_while_cond_1058917___redundant_placeholder25
1while_while_cond_1058917___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
ÿ?

D__inference_lstm_28_layer_call_and_return_conditional_losses_1059189

inputs(
lstm_cell_28_1059107:
ª¨(
lstm_cell_28_1059109:
ª¨#
lstm_cell_28_1059111:	¨
identity¢$lstm_cell_28/StatefulPartitionedCall¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¤
$lstm_cell_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_28_1059107lstm_cell_28_1059109lstm_cell_28_1059111*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_10590502&
$lstm_cell_28/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_28_1059107lstm_cell_28_1059109lstm_cell_28_1059111*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1059120*
condR
while_cond_1059119*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

Identity}
NoOpNoOp%^lstm_cell_28/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª: : : 2L
$lstm_cell_28/StatefulPartitionedCall$lstm_cell_28/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
èJ
Õ

lstm_27_while_body_1061140,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3+
'lstm_27_while_lstm_27_strided_slice_1_0g
clstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:
ª¨Q
=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ª¨K
<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	¨
lstm_27_while_identity
lstm_27_while_identity_1
lstm_27_while_identity_2
lstm_27_while_identity_3
lstm_27_while_identity_4
lstm_27_while_identity_5)
%lstm_27_while_lstm_27_strided_slice_1e
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorM
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:
ª¨O
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨I
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	¨¢1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp¢0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp¢2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpÓ
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2A
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0lstm_27_while_placeholderHlstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype023
1lstm_27/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype022
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp÷
!lstm_27/while/lstm_cell_27/MatMulMatMul8lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2#
!lstm_27/while/lstm_cell_27/MatMulè
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype024
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpà
#lstm_27/while/lstm_cell_27/MatMul_1MatMullstm_27_while_placeholder_2:lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2%
#lstm_27/while/lstm_cell_27/MatMul_1Ø
lstm_27/while/lstm_cell_27/addAddV2+lstm_27/while/lstm_cell_27/MatMul:product:0-lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2 
lstm_27/while/lstm_cell_27/addà
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype023
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpå
"lstm_27/while/lstm_cell_27/BiasAddBiasAdd"lstm_27/while/lstm_cell_27/add:z:09lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2$
"lstm_27/while/lstm_cell_27/BiasAdd
*lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_27/while/lstm_cell_27/split/split_dim¯
 lstm_27/while/lstm_cell_27/splitSplit3lstm_27/while/lstm_cell_27/split/split_dim:output:0+lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2"
 lstm_27/while/lstm_cell_27/split±
"lstm_27/while/lstm_cell_27/SigmoidSigmoid)lstm_27/while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2$
"lstm_27/while/lstm_cell_27/Sigmoidµ
$lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid)lstm_27/while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_27/while/lstm_cell_27/Sigmoid_1Á
lstm_27/while/lstm_cell_27/mulMul(lstm_27/while/lstm_cell_27/Sigmoid_1:y:0lstm_27_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2 
lstm_27/while/lstm_cell_27/mul¨
lstm_27/while/lstm_cell_27/ReluRelu)lstm_27/while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2!
lstm_27/while/lstm_cell_27/ReluÕ
 lstm_27/while/lstm_cell_27/mul_1Mul&lstm_27/while/lstm_cell_27/Sigmoid:y:0-lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_27/while/lstm_cell_27/mul_1Ê
 lstm_27/while/lstm_cell_27/add_1AddV2"lstm_27/while/lstm_cell_27/mul:z:0$lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_27/while/lstm_cell_27/add_1µ
$lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid)lstm_27/while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2&
$lstm_27/while/lstm_cell_27/Sigmoid_2§
!lstm_27/while/lstm_cell_27/Relu_1Relu$lstm_27/while/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2#
!lstm_27/while/lstm_cell_27/Relu_1Ù
 lstm_27/while/lstm_cell_27/mul_2Mul(lstm_27/while/lstm_cell_27/Sigmoid_2:y:0/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2"
 lstm_27/while/lstm_cell_27/mul_2
2lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_27_while_placeholder_1lstm_27_while_placeholder$lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_27/while/TensorArrayV2Write/TensorListSetIteml
lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_27/while/add/y
lstm_27/while/addAddV2lstm_27_while_placeholderlstm_27/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_27/while/addp
lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_27/while/add_1/y
lstm_27/while/add_1AddV2(lstm_27_while_lstm_27_while_loop_counterlstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_27/while/add_1
lstm_27/while/IdentityIdentitylstm_27/while/add_1:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity¦
lstm_27/while/Identity_1Identity.lstm_27_while_lstm_27_while_maximum_iterations^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_1
lstm_27/while/Identity_2Identitylstm_27/while/add:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_2º
lstm_27/while/Identity_3IdentityBlstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_3®
lstm_27/while/Identity_4Identity$lstm_27/while/lstm_cell_27/mul_2:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/while/Identity_4®
lstm_27/while/Identity_5Identity$lstm_27/while/lstm_cell_27/add_1:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_27/while/Identity_5
lstm_27/while/NoOpNoOp2^lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1^lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp3^lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_27/while/NoOp"9
lstm_27_while_identitylstm_27/while/Identity:output:0"=
lstm_27_while_identity_1!lstm_27/while/Identity_1:output:0"=
lstm_27_while_identity_2!lstm_27/while/Identity_2:output:0"=
lstm_27_while_identity_3!lstm_27/while/Identity_3:output:0"=
lstm_27_while_identity_4!lstm_27/while/Identity_4:output:0"=
lstm_27_while_identity_5!lstm_27/while/Identity_5:output:0"P
%lstm_27_while_lstm_27_strided_slice_1'lstm_27_while_lstm_27_strided_slice_1_0"z
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"|
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"x
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"È
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2f
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2d
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2h
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
¹
e
,__inference_dropout_46_layer_call_fn_1063904

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
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_46_layer_call_and_return_conditional_losses_10600532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
ÃU

D__inference_lstm_28_layer_call_and_return_conditional_losses_1060245

inputs?
+lstm_cell_28_matmul_readvariableop_resource:
ª¨A
-lstm_cell_28_matmul_1_readvariableop_resource:
ª¨;
,lstm_cell_28_biasadd_readvariableop_resource:	¨
identity¢#lstm_cell_28/BiasAdd/ReadVariableOp¢"lstm_cell_28/MatMul/ReadVariableOp¢$lstm_cell_28/MatMul_1/ReadVariableOp¢whileD
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
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2
zerosg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ª2
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
:ÿÿÿÿÿÿÿÿÿª2	
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
:ÿÿÿÿÿÿÿÿÿª2
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
valueB"ÿÿÿÿª   27
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
:ÿÿÿÿÿÿÿÿÿª*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_28/MatMul/ReadVariableOpReadVariableOp+lstm_cell_28_matmul_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02$
"lstm_cell_28/MatMul/ReadVariableOp­
lstm_cell_28/MatMulMatMulstrided_slice_2:output:0*lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul¼
$lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_28_matmul_1_readvariableop_resource* 
_output_shapes
:
ª¨*
dtype02&
$lstm_cell_28/MatMul_1/ReadVariableOp©
lstm_cell_28/MatMul_1MatMulzeros:output:0,lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/MatMul_1 
lstm_cell_28/addAddV2lstm_cell_28/MatMul:product:0lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/add´
#lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_28_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype02%
#lstm_cell_28/BiasAdd/ReadVariableOp­
lstm_cell_28/BiasAddBiasAddlstm_cell_28/add:z:0+lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
lstm_cell_28/BiasAdd~
lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_28/split/split_dim÷
lstm_cell_28/splitSplit%lstm_cell_28/split/split_dim:output:0lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
lstm_cell_28/split
lstm_cell_28/SigmoidSigmoidlstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid
lstm_cell_28/Sigmoid_1Sigmoidlstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_1
lstm_cell_28/mulMullstm_cell_28/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul~
lstm_cell_28/ReluRelulstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu
lstm_cell_28/mul_1Mullstm_cell_28/Sigmoid:y:0lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_1
lstm_cell_28/add_1AddV2lstm_cell_28/mul:z:0lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/add_1
lstm_cell_28/Sigmoid_2Sigmoidlstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Sigmoid_2}
lstm_cell_28/Relu_1Relulstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/Relu_1¡
lstm_cell_28/mul_2Mullstm_cell_28/Sigmoid_2:y:0!lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
lstm_cell_28/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_28_matmul_readvariableop_resource-lstm_cell_28_matmul_1_readvariableop_resource,lstm_cell_28_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1060161*
condR
while_cond_1060160*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª*
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
:ÿÿÿÿÿÿÿÿÿª2
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
:ÿÿÿÿÿÿÿÿÿª2

IdentityÈ
NoOpNoOp$^lstm_cell_28/BiasAdd/ReadVariableOp#^lstm_cell_28/MatMul/ReadVariableOp%^lstm_cell_28/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿª: : : 2J
#lstm_cell_28/BiasAdd/ReadVariableOp#lstm_cell_28/BiasAdd/ReadVariableOp2H
"lstm_cell_28/MatMul/ReadVariableOp"lstm_cell_28/MatMul/ReadVariableOp2L
$lstm_cell_28/MatMul_1/ReadVariableOp$lstm_cell_28/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs

e
G__inference_dropout_43_layer_call_and_return_conditional_losses_1059597

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿª:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
 
_user_specified_nameinputs
Æ1
ù
J__inference_sequential_12_layer_call_and_return_conditional_losses_1060840
lstm_26_input"
lstm_26_1060804:	¨#
lstm_26_1060806:
ª¨
lstm_26_1060808:	¨#
lstm_27_1060812:
ª¨#
lstm_27_1060814:
ª¨
lstm_27_1060816:	¨#
lstm_28_1060820:
ª¨#
lstm_28_1060822:
ª¨
lstm_28_1060824:	¨$
dense_29_1060828:
ªª
dense_29_1060830:	ª#
dense_30_1060834:	ª
dense_30_1060836:
identity¢ dense_29/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢"dropout_43/StatefulPartitionedCall¢"dropout_44/StatefulPartitionedCall¢"dropout_45/StatefulPartitionedCall¢"dropout_46/StatefulPartitionedCall¢lstm_26/StatefulPartitionedCall¢lstm_27/StatefulPartitionedCall¢lstm_28/StatefulPartitionedCall±
lstm_26/StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputlstm_26_1060804lstm_26_1060806lstm_26_1060808*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_26_layer_call_and_return_conditional_losses_10606212!
lstm_26/StatefulPartitionedCall
"dropout_43/StatefulPartitionedCallStatefulPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_43_layer_call_and_return_conditional_losses_10604622$
"dropout_43/StatefulPartitionedCallÏ
lstm_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_43/StatefulPartitionedCall:output:0lstm_27_1060812lstm_27_1060814lstm_27_1060816*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_27_layer_call_and_return_conditional_losses_10604332!
lstm_27/StatefulPartitionedCall¿
"dropout_44/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0#^dropout_43/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_44_layer_call_and_return_conditional_losses_10602742$
"dropout_44/StatefulPartitionedCallÏ
lstm_28/StatefulPartitionedCallStatefulPartitionedCall+dropout_44/StatefulPartitionedCall:output:0lstm_28_1060820lstm_28_1060822lstm_28_1060824*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_28_layer_call_and_return_conditional_losses_10602452!
lstm_28/StatefulPartitionedCall¿
"dropout_45/StatefulPartitionedCallStatefulPartitionedCall(lstm_28/StatefulPartitionedCall:output:0#^dropout_44/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_45_layer_call_and_return_conditional_losses_10600862$
"dropout_45/StatefulPartitionedCallÁ
 dense_29/StatefulPartitionedCallStatefulPartitionedCall+dropout_45/StatefulPartitionedCall:output:0dense_29_1060828dense_29_1060830*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_10599442"
 dense_29/StatefulPartitionedCallÀ
"dropout_46/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0#^dropout_45/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿª* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_46_layer_call_and_return_conditional_losses_10600532$
"dropout_46/StatefulPartitionedCallÀ
 dense_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_46/StatefulPartitionedCall:output:0dense_30_1060834dense_30_1060836*
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
E__inference_dense_30_layer_call_and_return_conditional_losses_10599872"
 dense_30/StatefulPartitionedCall
IdentityIdentity)dense_30/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall#^dropout_43/StatefulPartitionedCall#^dropout_44/StatefulPartitionedCall#^dropout_45/StatefulPartitionedCall#^dropout_46/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall ^lstm_28/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : 2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2H
"dropout_43/StatefulPartitionedCall"dropout_43/StatefulPartitionedCall2H
"dropout_44/StatefulPartitionedCall"dropout_44/StatefulPartitionedCall2H
"dropout_45/StatefulPartitionedCall"dropout_45/StatefulPartitionedCall2H
"dropout_46/StatefulPartitionedCall"dropout_46/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall2B
lstm_28/StatefulPartitionedCalllstm_28/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namelstm_26_input
³?
Õ
while_body_1062814
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_27_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_27/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
ô%
ì
while_body_1057722
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_26_1057746_0:	¨0
while_lstm_cell_26_1057748_0:
ª¨+
while_lstm_cell_26_1057750_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_26_1057746:	¨.
while_lstm_cell_26_1057748:
ª¨)
while_lstm_cell_26_1057750:	¨¢*while/lstm_cell_26/StatefulPartitionedCallÃ
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
*while/lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_26_1057746_0while_lstm_cell_26_1057748_0while_lstm_cell_26_1057750_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_10577082,
*while/lstm_cell_26/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_26/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_26/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_26/StatefulPartitionedCall*"
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
while_lstm_cell_26_1057746while_lstm_cell_26_1057746_0":
while_lstm_cell_26_1057748while_lstm_cell_26_1057748_0":
while_lstm_cell_26_1057750while_lstm_cell_26_1057750_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2X
*while/lstm_cell_26/StatefulPartitionedCall*while/lstm_cell_26/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Þ
È
while_cond_1063456
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1063456___redundant_placeholder05
1while_while_cond_1063456___redundant_placeholder15
1while_while_cond_1063456___redundant_placeholder25
1while_while_cond_1063456___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
Þ
È
while_cond_1063742
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1063742___redundant_placeholder05
1while_while_cond_1063742___redundant_placeholder15
1while_while_cond_1063742___redundant_placeholder25
1while_while_cond_1063742___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:
³?
Õ
while_body_1063743
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_28_matmul_readvariableop_resource_0:
ª¨I
5while_lstm_cell_28_matmul_1_readvariableop_resource_0:
ª¨C
4while_lstm_cell_28_biasadd_readvariableop_resource_0:	¨
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_28_matmul_readvariableop_resource:
ª¨G
3while_lstm_cell_28_matmul_1_readvariableop_resource:
ª¨A
2while_lstm_cell_28_biasadd_readvariableop_resource:	¨¢)while/lstm_cell_28/BiasAdd/ReadVariableOp¢(while/lstm_cell_28/MatMul/ReadVariableOp¢*while/lstm_cell_28/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿª   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_28/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_28_matmul_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02*
(while/lstm_cell_28/MatMul/ReadVariableOp×
while/lstm_cell_28/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMulÐ
*while/lstm_cell_28/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_28_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ª¨*
dtype02,
*while/lstm_cell_28/MatMul_1/ReadVariableOpÀ
while/lstm_cell_28/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/MatMul_1¸
while/lstm_cell_28/addAddV2#while/lstm_cell_28/MatMul:product:0%while/lstm_cell_28/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/addÈ
)while/lstm_cell_28/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_28_biasadd_readvariableop_resource_0*
_output_shapes	
:¨*
dtype02+
)while/lstm_cell_28/BiasAdd/ReadVariableOpÅ
while/lstm_cell_28/BiasAddBiasAddwhile/lstm_cell_28/add:z:01while/lstm_cell_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨2
while/lstm_cell_28/BiasAdd
"while/lstm_cell_28/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_28/split/split_dim
while/lstm_cell_28/splitSplit+while/lstm_cell_28/split/split_dim:output:0#while/lstm_cell_28/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª*
	num_split2
while/lstm_cell_28/split
while/lstm_cell_28/SigmoidSigmoid!while/lstm_cell_28/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid
while/lstm_cell_28/Sigmoid_1Sigmoid!while/lstm_cell_28/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_1¡
while/lstm_cell_28/mulMul while/lstm_cell_28/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul
while/lstm_cell_28/ReluRelu!while/lstm_cell_28/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Reluµ
while/lstm_cell_28/mul_1Mulwhile/lstm_cell_28/Sigmoid:y:0%while/lstm_cell_28/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_1ª
while/lstm_cell_28/add_1AddV2while/lstm_cell_28/mul:z:0while/lstm_cell_28/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/add_1
while/lstm_cell_28/Sigmoid_2Sigmoid!while/lstm_cell_28/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Sigmoid_2
while/lstm_cell_28/Relu_1Reluwhile/lstm_cell_28/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/Relu_1¹
while/lstm_cell_28/mul_2Mul while/lstm_cell_28/Sigmoid_2:y:0'while/lstm_cell_28/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/lstm_cell_28/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_28/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_28/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_28/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_28/BiasAdd/ReadVariableOp)^while/lstm_cell_28/MatMul/ReadVariableOp+^while/lstm_cell_28/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_28_biasadd_readvariableop_resource4while_lstm_cell_28_biasadd_readvariableop_resource_0"l
3while_lstm_cell_28_matmul_1_readvariableop_resource5while_lstm_cell_28_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_28_matmul_readvariableop_resource3while_lstm_cell_28_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: : : : : 2V
)while/lstm_cell_28/BiasAdd/ReadVariableOp)while/lstm_cell_28/BiasAdd/ReadVariableOp2T
(while/lstm_cell_28/MatMul/ReadVariableOp(while/lstm_cell_28/MatMul/ReadVariableOp2X
*while/lstm_cell_28/MatMul_1/ReadVariableOp*while/lstm_cell_28/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
: 
Þ
¹
)__inference_lstm_26_layer_call_fn_1061936
inputs_0
unknown:	¨
	unknown_0:
ª¨
	unknown_1:	¨
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_26_layer_call_and_return_conditional_losses_10577912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª2

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
while_cond_1057721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1057721___redundant_placeholder05
1while_while_cond_1057721___redundant_placeholder15
1while_while_cond_1057721___redundant_placeholder25
1while_while_cond_1057721___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿª:ÿÿÿÿÿÿÿÿÿª: ::::: 
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
:ÿÿÿÿÿÿÿÿÿª:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª:

_output_shapes
: :

_output_shapes
:"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
K
lstm_26_input:
serving_default_lstm_26_input:0ÿÿÿÿÿÿÿÿÿ@
dense_304
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
	variables
regularization_losses
trainable_variables
	keras_api

signatures
Ä_default_save_signature
Å__call__
+Æ&call_and_return_all_conditional_losses"
_tf_keras_sequential
Å
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
	variables
regularization_losses
trainable_variables
	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"
_tf_keras_layer
Å
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
 	variables
!regularization_losses
"trainable_variables
#	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"
_tf_keras_layer
Å
$cell
%
state_spec
&	variables
'regularization_losses
(trainable_variables
)	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
*	variables
+regularization_losses
,trainable_variables
-	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"
_tf_keras_layer
½

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"
_tf_keras_layer
§
4	variables
5regularization_losses
6trainable_variables
7	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
½

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
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
Î
	variables
regularization_losses
Lmetrics
Mlayer_metrics

Nlayers
Olayer_regularization_losses
trainable_variables
Pnon_trainable_variables
Å__call__
Ä_default_save_signature
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
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
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
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
¼
	variables
regularization_losses
Vmetrics
Wlayer_metrics

Xlayers
Ylayer_regularization_losses
trainable_variables
Znon_trainable_variables

[states
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
	variables
regularization_losses
\metrics
]layer_metrics

^layers
_layer_regularization_losses
trainable_variables
`non_trainable_variables
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
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"
_tf_keras_layer
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
¼
	variables
regularization_losses
fmetrics
glayer_metrics

hlayers
ilayer_regularization_losses
trainable_variables
jnon_trainable_variables

kstates
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
 	variables
!regularization_losses
lmetrics
mlayer_metrics

nlayers
olayer_regularization_losses
"trainable_variables
pnon_trainable_variables
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
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
Þ__call__
+ß&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
¼
&	variables
'regularization_losses
vmetrics
wlayer_metrics

xlayers
ylayer_regularization_losses
(trainable_variables
znon_trainable_variables

{states
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
*	variables
+regularization_losses
|metrics
}layer_metrics

~layers
layer_regularization_losses
,trainable_variables
non_trainable_variables
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
#:!
ªª2dense_29/kernel
:ª2dense_29/bias
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
µ
0	variables
1regularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
2trainable_variables
non_trainable_variables
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
4	variables
5regularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
6trainable_variables
non_trainable_variables
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
": 	ª2dense_30/kernel
:2dense_30/bias
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
µ
:	variables
;regularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
<trainable_variables
non_trainable_variables
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
.:,	¨2lstm_26/lstm_cell_26/kernel
9:7
ª¨2%lstm_26/lstm_cell_26/recurrent_kernel
(:&¨2lstm_26/lstm_cell_26/bias
/:-
ª¨2lstm_27/lstm_cell_27/kernel
9:7
ª¨2%lstm_27/lstm_cell_27/recurrent_kernel
(:&¨2lstm_27/lstm_cell_27/bias
/:-
ª¨2lstm_28/lstm_cell_28/kernel
9:7
ª¨2%lstm_28/lstm_cell_28/recurrent_kernel
(:&¨2lstm_28/lstm_cell_28/bias
0
0
1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
µ
R	variables
Sregularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
Ttrainable_variables
non_trainable_variables
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
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
µ
b	variables
cregularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
dtrainable_variables
non_trainable_variables
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
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
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
µ
r	variables
sregularization_losses
metrics
layer_metrics
layers
 layer_regularization_losses
ttrainable_variables
 non_trainable_variables
Þ__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
$0"
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
ªª2Adam/dense_29/kernel/m
!:ª2Adam/dense_29/bias/m
':%	ª2Adam/dense_30/kernel/m
 :2Adam/dense_30/bias/m
3:1	¨2"Adam/lstm_26/lstm_cell_26/kernel/m
>:<
ª¨2,Adam/lstm_26/lstm_cell_26/recurrent_kernel/m
-:+¨2 Adam/lstm_26/lstm_cell_26/bias/m
4:2
ª¨2"Adam/lstm_27/lstm_cell_27/kernel/m
>:<
ª¨2,Adam/lstm_27/lstm_cell_27/recurrent_kernel/m
-:+¨2 Adam/lstm_27/lstm_cell_27/bias/m
4:2
ª¨2"Adam/lstm_28/lstm_cell_28/kernel/m
>:<
ª¨2,Adam/lstm_28/lstm_cell_28/recurrent_kernel/m
-:+¨2 Adam/lstm_28/lstm_cell_28/bias/m
(:&
ªª2Adam/dense_29/kernel/v
!:ª2Adam/dense_29/bias/v
':%	ª2Adam/dense_30/kernel/v
 :2Adam/dense_30/bias/v
3:1	¨2"Adam/lstm_26/lstm_cell_26/kernel/v
>:<
ª¨2,Adam/lstm_26/lstm_cell_26/recurrent_kernel/v
-:+¨2 Adam/lstm_26/lstm_cell_26/bias/v
4:2
ª¨2"Adam/lstm_27/lstm_cell_27/kernel/v
>:<
ª¨2,Adam/lstm_27/lstm_cell_27/recurrent_kernel/v
-:+¨2 Adam/lstm_27/lstm_cell_27/bias/v
4:2
ª¨2"Adam/lstm_28/lstm_cell_28/kernel/v
>:<
ª¨2,Adam/lstm_28/lstm_cell_28/recurrent_kernel/v
-:+¨2 Adam/lstm_28/lstm_cell_28/bias/v
ÓBÐ
"__inference__wrapped_model_1057641lstm_26_input"
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
2
/__inference_sequential_12_layer_call_fn_1060023
/__inference_sequential_12_layer_call_fn_1060910
/__inference_sequential_12_layer_call_fn_1060941
/__inference_sequential_12_layer_call_fn_1060762À
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
J__inference_sequential_12_layer_call_and_return_conditional_losses_1061419
J__inference_sequential_12_layer_call_and_return_conditional_losses_1061925
J__inference_sequential_12_layer_call_and_return_conditional_losses_1060801
J__inference_sequential_12_layer_call_and_return_conditional_losses_1060840À
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
2
)__inference_lstm_26_layer_call_fn_1061936
)__inference_lstm_26_layer_call_fn_1061947
)__inference_lstm_26_layer_call_fn_1061958
)__inference_lstm_26_layer_call_fn_1061969Õ
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
D__inference_lstm_26_layer_call_and_return_conditional_losses_1062112
D__inference_lstm_26_layer_call_and_return_conditional_losses_1062255
D__inference_lstm_26_layer_call_and_return_conditional_losses_1062398
D__inference_lstm_26_layer_call_and_return_conditional_losses_1062541Õ
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
,__inference_dropout_43_layer_call_fn_1062546
,__inference_dropout_43_layer_call_fn_1062551´
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
G__inference_dropout_43_layer_call_and_return_conditional_losses_1062556
G__inference_dropout_43_layer_call_and_return_conditional_losses_1062568´
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
)__inference_lstm_27_layer_call_fn_1062579
)__inference_lstm_27_layer_call_fn_1062590
)__inference_lstm_27_layer_call_fn_1062601
)__inference_lstm_27_layer_call_fn_1062612Õ
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
D__inference_lstm_27_layer_call_and_return_conditional_losses_1062755
D__inference_lstm_27_layer_call_and_return_conditional_losses_1062898
D__inference_lstm_27_layer_call_and_return_conditional_losses_1063041
D__inference_lstm_27_layer_call_and_return_conditional_losses_1063184Õ
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
,__inference_dropout_44_layer_call_fn_1063189
,__inference_dropout_44_layer_call_fn_1063194´
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
G__inference_dropout_44_layer_call_and_return_conditional_losses_1063199
G__inference_dropout_44_layer_call_and_return_conditional_losses_1063211´
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
)__inference_lstm_28_layer_call_fn_1063222
)__inference_lstm_28_layer_call_fn_1063233
)__inference_lstm_28_layer_call_fn_1063244
)__inference_lstm_28_layer_call_fn_1063255Õ
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
D__inference_lstm_28_layer_call_and_return_conditional_losses_1063398
D__inference_lstm_28_layer_call_and_return_conditional_losses_1063541
D__inference_lstm_28_layer_call_and_return_conditional_losses_1063684
D__inference_lstm_28_layer_call_and_return_conditional_losses_1063827Õ
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
,__inference_dropout_45_layer_call_fn_1063832
,__inference_dropout_45_layer_call_fn_1063837´
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
G__inference_dropout_45_layer_call_and_return_conditional_losses_1063842
G__inference_dropout_45_layer_call_and_return_conditional_losses_1063854´
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
*__inference_dense_29_layer_call_fn_1063863¢
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
E__inference_dense_29_layer_call_and_return_conditional_losses_1063894¢
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
,__inference_dropout_46_layer_call_fn_1063899
,__inference_dropout_46_layer_call_fn_1063904´
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
G__inference_dropout_46_layer_call_and_return_conditional_losses_1063909
G__inference_dropout_46_layer_call_and_return_conditional_losses_1063921´
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
*__inference_dense_30_layer_call_fn_1063930¢
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
E__inference_dense_30_layer_call_and_return_conditional_losses_1063960¢
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
%__inference_signature_wrapper_1060879lstm_26_input"
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
.__inference_lstm_cell_26_layer_call_fn_1063977
.__inference_lstm_cell_26_layer_call_fn_1063994¾
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
I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_1064026
I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_1064058¾
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
.__inference_lstm_cell_27_layer_call_fn_1064075
.__inference_lstm_cell_27_layer_call_fn_1064092¾
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
I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_1064124
I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_1064156¾
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
.__inference_lstm_cell_28_layer_call_fn_1064173
.__inference_lstm_cell_28_layer_call_fn_1064190¾
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
I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_1064222
I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_1064254¾
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
"__inference__wrapped_model_1057641CDEFGHIJK./89:¢7
0¢-
+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2
dense_30&#
dense_30ÿÿÿÿÿÿÿÿÿ¯
E__inference_dense_29_layer_call_and_return_conditional_losses_1063894f./4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿª
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 
*__inference_dense_29_layer_call_fn_1063863Y./4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿª
ª "ÿÿÿÿÿÿÿÿÿª®
E__inference_dense_30_layer_call_and_return_conditional_losses_1063960e894¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿª
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_30_layer_call_fn_1063930X894¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿª
ª "ÿÿÿÿÿÿÿÿÿ±
G__inference_dropout_43_layer_call_and_return_conditional_losses_1062556f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 ±
G__inference_dropout_43_layer_call_and_return_conditional_losses_1062568f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 
,__inference_dropout_43_layer_call_fn_1062546Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p 
ª "ÿÿÿÿÿÿÿÿÿª
,__inference_dropout_43_layer_call_fn_1062551Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p
ª "ÿÿÿÿÿÿÿÿÿª±
G__inference_dropout_44_layer_call_and_return_conditional_losses_1063199f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 ±
G__inference_dropout_44_layer_call_and_return_conditional_losses_1063211f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 
,__inference_dropout_44_layer_call_fn_1063189Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p 
ª "ÿÿÿÿÿÿÿÿÿª
,__inference_dropout_44_layer_call_fn_1063194Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p
ª "ÿÿÿÿÿÿÿÿÿª±
G__inference_dropout_45_layer_call_and_return_conditional_losses_1063842f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 ±
G__inference_dropout_45_layer_call_and_return_conditional_losses_1063854f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 
,__inference_dropout_45_layer_call_fn_1063832Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p 
ª "ÿÿÿÿÿÿÿÿÿª
,__inference_dropout_45_layer_call_fn_1063837Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p
ª "ÿÿÿÿÿÿÿÿÿª±
G__inference_dropout_46_layer_call_and_return_conditional_losses_1063909f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 ±
G__inference_dropout_46_layer_call_and_return_conditional_losses_1063921f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 
,__inference_dropout_46_layer_call_fn_1063899Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p 
ª "ÿÿÿÿÿÿÿÿÿª
,__inference_dropout_46_layer_call_fn_1063904Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿª
p
ª "ÿÿÿÿÿÿÿÿÿªÔ
D__inference_lstm_26_layer_call_and_return_conditional_losses_1062112CDEO¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 Ô
D__inference_lstm_26_layer_call_and_return_conditional_losses_1062255CDEO¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 º
D__inference_lstm_26_layer_call_and_return_conditional_losses_1062398rCDE?¢<
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
0ÿÿÿÿÿÿÿÿÿª
 º
D__inference_lstm_26_layer_call_and_return_conditional_losses_1062541rCDE?¢<
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
0ÿÿÿÿÿÿÿÿÿª
 «
)__inference_lstm_26_layer_call_fn_1061936~CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª«
)__inference_lstm_26_layer_call_fn_1061947~CDEO¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
)__inference_lstm_26_layer_call_fn_1061958eCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿª
)__inference_lstm_26_layer_call_fn_1061969eCDE?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿªÕ
D__inference_lstm_27_layer_call_and_return_conditional_losses_1062755FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 Õ
D__inference_lstm_27_layer_call_and_return_conditional_losses_1062898FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 »
D__inference_lstm_27_layer_call_and_return_conditional_losses_1063041sFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿª

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 »
D__inference_lstm_27_layer_call_and_return_conditional_losses_1063184sFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿª

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 ¬
)__inference_lstm_27_layer_call_fn_1062579FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª¬
)__inference_lstm_27_layer_call_fn_1062590FGHP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
)__inference_lstm_27_layer_call_fn_1062601fFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿª

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿª
)__inference_lstm_27_layer_call_fn_1062612fFGH@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿª

 
p

 
ª "ÿÿÿÿÿÿÿÿÿªÕ
D__inference_lstm_28_layer_call_and_return_conditional_losses_1063398IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 Õ
D__inference_lstm_28_layer_call_and_return_conditional_losses_1063541IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
 »
D__inference_lstm_28_layer_call_and_return_conditional_losses_1063684sIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿª

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 »
D__inference_lstm_28_layer_call_and_return_conditional_losses_1063827sIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿª

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿª
 ¬
)__inference_lstm_28_layer_call_fn_1063222IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª¬
)__inference_lstm_28_layer_call_fn_1063233IJKP¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿª
)__inference_lstm_28_layer_call_fn_1063244fIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿª

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿª
)__inference_lstm_28_layer_call_fn_1063255fIJK@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿª

 
p

 
ª "ÿÿÿÿÿÿÿÿÿªÐ
I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_1064026CDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿª
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿª
 
0/1/1ÿÿÿÿÿÿÿÿÿª
 Ð
I__inference_lstm_cell_26_layer_call_and_return_conditional_losses_1064058CDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿª
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿª
 
0/1/1ÿÿÿÿÿÿÿÿÿª
 ¥
.__inference_lstm_cell_26_layer_call_fn_1063977òCDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿª
C@

1/0ÿÿÿÿÿÿÿÿÿª

1/1ÿÿÿÿÿÿÿÿÿª¥
.__inference_lstm_cell_26_layer_call_fn_1063994òCDE¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿª
C@

1/0ÿÿÿÿÿÿÿÿÿª

1/1ÿÿÿÿÿÿÿÿÿªÒ
I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_1064124FGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿª
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿª
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿª
 
0/1/1ÿÿÿÿÿÿÿÿÿª
 Ò
I__inference_lstm_cell_27_layer_call_and_return_conditional_losses_1064156FGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿª
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿª
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿª
 
0/1/1ÿÿÿÿÿÿÿÿÿª
 §
.__inference_lstm_cell_27_layer_call_fn_1064075ôFGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿª
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿª
C@

1/0ÿÿÿÿÿÿÿÿÿª

1/1ÿÿÿÿÿÿÿÿÿª§
.__inference_lstm_cell_27_layer_call_fn_1064092ôFGH¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿª
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿª
C@

1/0ÿÿÿÿÿÿÿÿÿª

1/1ÿÿÿÿÿÿÿÿÿªÒ
I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_1064222IJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿª
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿª
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿª
 
0/1/1ÿÿÿÿÿÿÿÿÿª
 Ò
I__inference_lstm_cell_28_layer_call_and_return_conditional_losses_1064254IJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿª
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿª
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿª
 
0/1/1ÿÿÿÿÿÿÿÿÿª
 §
.__inference_lstm_cell_28_layer_call_fn_1064173ôIJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿª
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿª
C@

1/0ÿÿÿÿÿÿÿÿÿª

1/1ÿÿÿÿÿÿÿÿÿª§
.__inference_lstm_cell_28_layer_call_fn_1064190ôIJK¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿª
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿª
# 
states/1ÿÿÿÿÿÿÿÿÿª
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿª
C@

1/0ÿÿÿÿÿÿÿÿÿª

1/1ÿÿÿÿÿÿÿÿÿªÌ
J__inference_sequential_12_layer_call_and_return_conditional_losses_1060801~CDEFGHIJK./89B¢?
8¢5
+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ì
J__inference_sequential_12_layer_call_and_return_conditional_losses_1060840~CDEFGHIJK./89B¢?
8¢5
+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Å
J__inference_sequential_12_layer_call_and_return_conditional_losses_1061419wCDEFGHIJK./89;¢8
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
J__inference_sequential_12_layer_call_and_return_conditional_losses_1061925wCDEFGHIJK./89;¢8
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
/__inference_sequential_12_layer_call_fn_1060023qCDEFGHIJK./89B¢?
8¢5
+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¤
/__inference_sequential_12_layer_call_fn_1060762qCDEFGHIJK./89B¢?
8¢5
+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_12_layer_call_fn_1060910jCDEFGHIJK./89;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_12_layer_call_fn_1060941jCDEFGHIJK./89;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¿
%__inference_signature_wrapper_1060879CDEFGHIJK./89K¢H
¢ 
Aª>
<
lstm_26_input+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ"7ª4
2
dense_30&#
dense_30ÿÿÿÿÿÿÿÿÿ