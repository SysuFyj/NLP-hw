��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8մ
�
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d2*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	�d2*
dtype0
�
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(2*'
shared_nameembedding_1/embeddings
�
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes

:(2*
dtype0
�
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*'
shared_nameembedding_2/embeddings
�
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes
:	�2*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�d*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:d*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	d�*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
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
�
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d2*,
shared_nameAdam/embedding/embeddings/m
�
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	�d2*
dtype0
�
Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(2*.
shared_nameAdam/embedding_1/embeddings/m
�
1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m*
_output_shapes

:(2*
dtype0
�
Adam/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*.
shared_nameAdam/embedding_2/embeddings/m
�
1Adam/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/m*
_output_shapes
:	�2*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	�d*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:d*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*&
shared_nameAdam/dense_1/kernel/m
�
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	d�*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d2*,
shared_nameAdam/embedding/embeddings/v
�
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	�d2*
dtype0
�
Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(2*.
shared_nameAdam/embedding_1/embeddings/v
�
1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v*
_output_shapes

:(2*
dtype0
�
Adam/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*.
shared_nameAdam/embedding_2/embeddings/v
�
1Adam/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/v*
_output_shapes
:	�2*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	�d*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:d*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*&
shared_nameAdam/dense_1/kernel/v
�
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	d�*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�:
value�:B�9 B�9
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
 
 
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
b

embeddings
regularization_losses
 	variables
!trainable_variables
"	keras_api
R
#regularization_losses
$	variables
%trainable_variables
&	keras_api
R
'regularization_losses
(	variables
)trainable_variables
*	keras_api
R
+regularization_losses
,	variables
-trainable_variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
R
5regularization_losses
6	variables
7trainable_variables
8	keras_api
R
9regularization_losses
:	variables
;trainable_variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
�
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem�m�m�/m�0m�=m�>m�v�v�v�/v�0v�=v�>v�
 
1
0
1
2
/3
04
=5
>6
1
0
1
2
/3
04
=5
>6
�
regularization_losses
Hlayer_metrics
	variables
Imetrics
Jnon_trainable_variables
trainable_variables
Klayer_regularization_losses

Llayers
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
�
Mlayer_metrics
regularization_losses
	variables
Nmetrics
Onon_trainable_variables
trainable_variables
Player_regularization_losses

Qlayers
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
�
Rlayer_metrics
regularization_losses
	variables
Smetrics
Tnon_trainable_variables
trainable_variables
Ulayer_regularization_losses

Vlayers
fd
VARIABLE_VALUEembedding_2/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
�
Wlayer_metrics
regularization_losses
 	variables
Xmetrics
Ynon_trainable_variables
!trainable_variables
Zlayer_regularization_losses

[layers
 
 
 
�
\layer_metrics
#regularization_losses
$	variables
]metrics
^non_trainable_variables
%trainable_variables
_layer_regularization_losses

`layers
 
 
 
�
alayer_metrics
'regularization_losses
(	variables
bmetrics
cnon_trainable_variables
)trainable_variables
dlayer_regularization_losses

elayers
 
 
 
�
flayer_metrics
+regularization_losses
,	variables
gmetrics
hnon_trainable_variables
-trainable_variables
ilayer_regularization_losses

jlayers
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
�
klayer_metrics
1regularization_losses
2	variables
lmetrics
mnon_trainable_variables
3trainable_variables
nlayer_regularization_losses

olayers
 
 
 
�
player_metrics
5regularization_losses
6	variables
qmetrics
rnon_trainable_variables
7trainable_variables
slayer_regularization_losses

tlayers
 
 
 
�
ulayer_metrics
9regularization_losses
:	variables
vmetrics
wnon_trainable_variables
;trainable_variables
xlayer_regularization_losses

ylayers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
�
zlayer_metrics
?regularization_losses
@	variables
{metrics
|non_trainable_variables
Atrainable_variables
}layer_regularization_losses

~layers
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
 

0
�1
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
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

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
��
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding_1/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding_2/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding_1/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/embedding_2/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
z
serving_default_input_2Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
z
serving_default_input_3Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3embedding/embeddingsembedding_1/embeddingsembedding_2/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2
*
Tout
2*(
_output_shapes
:����������*)
_read_only_resource_inputs
		*/
config_proto

GPU

CPU2 *0J 8*+
f&R$
"__inference_signature_wrapper_8084
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOp*embedding_2/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp1Adam/embedding_2/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp1Adam/embedding_2/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*+
Tin$
"2 	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*&
f!R
__inference__traced_save_8842
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsembedding_1/embeddingsembedding_2/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/embedding_1/embeddings/mAdam/embedding_2/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/embedding/embeddings/vAdam/embedding_1/embeddings/vAdam/embedding_2/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v**
Tin#
!2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*)
f$R"
 __inference__traced_restore_8944��
�

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_7579

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
L__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_8447
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*
_cloned(*+
_output_shapes
:���������022
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:���������022

Identity"
identityIdentity:output:0*X
_input_shapesG
E:���������2:���������2:���������2:U Q
+
_output_shapes
:���������2
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������2
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:���������2
"
_user_specified_name
inputs/2
�
`
A__inference_dropout_layer_call_and_return_conditional_losses_7493

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
A__inference_dropout_layer_call_and_return_conditional_losses_8477

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�w
�
?__inference_model_layer_call_and_return_conditional_losses_7782
input_1
input_2
input_3
embedding_7701
embedding_1_7704
embedding_2_7707

dense_7713

dense_7715
dense_1_7720
dense_1_7722
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_7701*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_73772#
!embedding/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_7704*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_74072%
#embedding_1/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_3embedding_2_7707*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_74372%
#embedding_2/StatefulPartitionedCall�
"tf_op_layer_concat/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:���������02* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_74572$
"tf_op_layer_concat/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_74732%
#tf_op_layer_Reshape/PartitionedCall�
dropout/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74982
dropout/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dense_7713
dense_7715*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_75372
dense/StatefulPartitionedCall�
tf_op_layer_Pow/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_75592!
tf_op_layer_Pow/PartitionedCall�
dropout_1/PartitionedCallPartitionedCall(tf_op_layer_Pow/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75842
dropout_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_7720dense_1_7722*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_76232!
dense_1/StatefulPartitionedCall�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_7701*
_output_shapes
:	�d2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_7704*
_output_shapes

:(2*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:(22+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_7707*
_output_shapes
:	�2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_7713*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp
dense_7715*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_7720*
_output_shapes
:	d�*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_7722*
_output_shapes	
:�*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/add�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�z
�
?__inference_model_layer_call_and_return_conditional_losses_7873

inputs
inputs_1
inputs_2
embedding_7792
embedding_1_7795
embedding_2_7798

dense_7804

dense_7806
dense_1_7811
dense_1_7813
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_7792*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_73772#
!embedding/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_1_7795*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_74072%
#embedding_1/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_2_7798*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_74372%
#embedding_2/StatefulPartitionedCall�
"tf_op_layer_concat/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:���������02* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_74572$
"tf_op_layer_concat/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_74732%
#tf_op_layer_Reshape/PartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74932!
dropout/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
dense_7804
dense_7806*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_75372
dense/StatefulPartitionedCall�
tf_op_layer_Pow/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_75592!
tf_op_layer_Pow/PartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Pow/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75792#
!dropout_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_7811dense_1_7813*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_76232!
dense_1/StatefulPartitionedCall�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_7792*
_output_shapes
:	�d2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_7795*
_output_shapes

:(2*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:(22+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_7798*
_output_shapes
:	�2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_7804*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp
dense_7806*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_7811*
_output_shapes
:	d�*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_7813*
_output_shapes	
:�*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/add�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :
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
: 
�z
�
?__inference_model_layer_call_and_return_conditional_losses_7696
input_1
input_2
input_3
embedding_7386
embedding_1_7416
embedding_2_7446

dense_7548

dense_7550
dense_1_7634
dense_1_7636
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_7386*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_73772#
!embedding/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_7416*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_74072%
#embedding_1/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_3embedding_2_7446*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_74372%
#embedding_2/StatefulPartitionedCall�
"tf_op_layer_concat/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:���������02* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_74572$
"tf_op_layer_concat/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_74732%
#tf_op_layer_Reshape/PartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74932!
dropout/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
dense_7548
dense_7550*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_75372
dense/StatefulPartitionedCall�
tf_op_layer_Pow/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_75592!
tf_op_layer_Pow/PartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Pow/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75792#
!dropout_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_7634dense_1_7636*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_76232!
dense_1/StatefulPartitionedCall�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_7386*
_output_shapes
:	�d2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_7416*
_output_shapes

:(2*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:(22+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_7446*
_output_shapes
:	�2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_7548*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp
dense_7550*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_7634*
_output_shapes
:	d�*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_7636*
_output_shapes	
:�*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/add�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_7623

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/adde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_8571

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
h
__inference_loss_fn_4_86979
5dense_bias_regularizer_square_readvariableop_resource
identity��
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp5dense_bias_regularizer_square_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/adda
IdentityIdentitydense/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�
�
$__inference_model_layer_call_fn_7997
input_1
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*(
_output_shapes
:����������*)
_read_only_resource_inputs
		*/
config_proto

GPU

CPU2 *0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_79802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�
D
(__inference_dropout_1_layer_call_fn_8581

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_7498

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�5
�
__inference__wrapped_model_7353
input_1
input_2
input_3)
%model_embedding_embedding_lookup_7315+
'model_embedding_1_embedding_lookup_7321+
'model_embedding_2_embedding_lookup_7327.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource
identity�~
model/embedding/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:���������2
model/embedding/Cast�
 model/embedding/embedding_lookupResourceGather%model_embedding_embedding_lookup_7315model/embedding/Cast:y:0*
Tindices0*8
_class.
,*loc:@model/embedding/embedding_lookup/7315*+
_output_shapes
:���������2*
dtype02"
 model/embedding/embedding_lookup�
)model/embedding/embedding_lookup/IdentityIdentity)model/embedding/embedding_lookup:output:0*
T0*8
_class.
,*loc:@model/embedding/embedding_lookup/7315*+
_output_shapes
:���������22+
)model/embedding/embedding_lookup/Identity�
+model/embedding/embedding_lookup/Identity_1Identity2model/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22-
+model/embedding/embedding_lookup/Identity_1�
model/embedding_1/CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:���������2
model/embedding_1/Cast�
"model/embedding_1/embedding_lookupResourceGather'model_embedding_1_embedding_lookup_7321model/embedding_1/Cast:y:0*
Tindices0*:
_class0
.,loc:@model/embedding_1/embedding_lookup/7321*+
_output_shapes
:���������2*
dtype02$
"model/embedding_1/embedding_lookup�
+model/embedding_1/embedding_lookup/IdentityIdentity+model/embedding_1/embedding_lookup:output:0*
T0*:
_class0
.,loc:@model/embedding_1/embedding_lookup/7321*+
_output_shapes
:���������22-
+model/embedding_1/embedding_lookup/Identity�
-model/embedding_1/embedding_lookup/Identity_1Identity4model/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22/
-model/embedding_1/embedding_lookup/Identity_1�
model/embedding_2/CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:���������2
model/embedding_2/Cast�
"model/embedding_2/embedding_lookupResourceGather'model_embedding_2_embedding_lookup_7327model/embedding_2/Cast:y:0*
Tindices0*:
_class0
.,loc:@model/embedding_2/embedding_lookup/7327*+
_output_shapes
:���������2*
dtype02$
"model/embedding_2/embedding_lookup�
+model/embedding_2/embedding_lookup/IdentityIdentity+model/embedding_2/embedding_lookup:output:0*
T0*:
_class0
.,loc:@model/embedding_2/embedding_lookup/7327*+
_output_shapes
:���������22-
+model/embedding_2/embedding_lookup/Identity�
-model/embedding_2/embedding_lookup/Identity_1Identity4model/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22/
-model/embedding_2/embedding_lookup/Identity_1�
$model/tf_op_layer_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$model/tf_op_layer_concat/concat/axis�
model/tf_op_layer_concat/concatConcatV24model/embedding/embedding_lookup/Identity_1:output:06model/embedding_1/embedding_lookup/Identity_1:output:06model/embedding_2/embedding_lookup/Identity_1:output:0-model/tf_op_layer_concat/concat/axis:output:0*
N*
T0*
_cloned(*+
_output_shapes
:���������022!
model/tf_op_layer_concat/concat�
'model/tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����`	  2)
'model/tf_op_layer_Reshape/Reshape/shape�
!model/tf_op_layer_Reshape/ReshapeReshape(model/tf_op_layer_concat/concat:output:00model/tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*(
_output_shapes
:����������2#
!model/tf_op_layer_Reshape/Reshape�
model/dropout/IdentityIdentity*model/tf_op_layer_Reshape/Reshape:output:0*
T0*(
_output_shapes
:����������2
model/dropout/Identity�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMulmodel/dropout/Identity:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
model/dense/BiasAdd
model/tf_op_layer_Pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
model/tf_op_layer_Pow/Pow/y�
model/tf_op_layer_Pow/PowPowmodel/dense/BiasAdd:output:0$model/tf_op_layer_Pow/Pow/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������d2
model/tf_op_layer_Pow/Pow�
model/dropout_1/IdentityIdentitymodel/tf_op_layer_Pow/Pow:z:0*
T0*'
_output_shapes
:���������d2
model/dropout_1/Identity�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02%
#model/dense_1/MatMul/ReadVariableOp�
model/dense_1/MatMulMatMul!model/dropout_1/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense_1/MatMul�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense_1/BiasAdds
IdentityIdentitymodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������::::::::P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�w
�
?__inference_model_layer_call_and_return_conditional_losses_7980

inputs
inputs_1
inputs_2
embedding_7899
embedding_1_7902
embedding_2_7905

dense_7911

dense_7913
dense_1_7918
dense_1_7920
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_7899*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_73772#
!embedding/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_1_7902*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_74072%
#embedding_1/StatefulPartitionedCall�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_2_7905*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_74372%
#embedding_2/StatefulPartitionedCall�
"tf_op_layer_concat/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:���������02* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_74572$
"tf_op_layer_concat/PartitionedCall�
#tf_op_layer_Reshape/PartitionedCallPartitionedCall+tf_op_layer_concat/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_74732%
#tf_op_layer_Reshape/PartitionedCall�
dropout/PartitionedCallPartitionedCall,tf_op_layer_Reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74982
dropout/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dense_7911
dense_7913*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_75372
dense/StatefulPartitionedCall�
tf_op_layer_Pow/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_75592!
tf_op_layer_Pow/PartitionedCall�
dropout_1/PartitionedCallPartitionedCall(tf_op_layer_Pow/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75842
dropout_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_7918dense_1_7920*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_76232!
dense_1/StatefulPartitionedCall�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_7899*
_output_shapes
:	�d2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_7902*
_output_shapes

:(2*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:(22+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_7905*
_output_shapes
:	�2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp
dense_7911*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp
dense_7913*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_7918*
_output_shapes
:	d�*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_7920*
_output_shapes	
:�*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/add�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :
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
: 
�
a
(__inference_dropout_1_layer_call_fn_8576

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_8340
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*(
_output_shapes
:����������*)
_read_only_resource_inputs
		*/
config_proto

GPU

CPU2 *0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_79802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :
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
: 
�
{
&__inference_dense_1_layer_call_fn_8632

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_76232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
y
$__inference_dense_layer_call_fn_8543

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������d*$
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_75372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
p
*__inference_embedding_2_layer_call_fn_8439

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_74372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
n
(__inference_embedding_layer_call_fn_8373

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_73772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
B
&__inference_dropout_layer_call_fn_8492

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74982
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
p
*__inference_embedding_1_layer_call_fn_8406

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:���������2*#
_read_only_resource_inputs
*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_74072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
t
__inference_loss_fn_1_8658E
Aembedding_1_embeddings_regularizer_square_readvariableop_resource
identity��
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_1_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:(2*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:(22+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/addm
IdentityIdentity*embedding_1/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�
i
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_8460

inputs
identityo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����`	  2
Reshape/shape
ReshapeReshapeinputsReshape/shape:output:0*
T0*
_cloned(*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������02:S O
+
_output_shapes
:���������02
 
_user_specified_nameinputs
�
�
$__inference_model_layer_call_fn_7890
input_1
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*(
_output_shapes
:����������*)
_read_only_resource_inputs
		*/
config_proto

GPU

CPU2 *0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_78732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�
�
$__inference_model_layer_call_fn_8319
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*(
_output_shapes
:����������*)
_read_only_resource_inputs
		*/
config_proto

GPU

CPU2 *0J 8*H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_78732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :
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
: 
�
l
__inference_loss_fn_5_8710=
9dense_1_kernel_regularizer_square_readvariableop_resource
identity��
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	d�*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/adde
IdentityIdentity"dense_1/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�
_
&__inference_dropout_layer_call_fn_8487

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_7473

inputs
identityo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����`	  2
Reshape/shape
ReshapeReshapeinputsReshape/shape:output:0*
T0*
_cloned(*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������02:S O
+
_output_shapes
:���������02
 
_user_specified_nameinputs
�
e
I__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_7559

inputs
identityS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/yj
PowPowinputsPow/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������d2
Pow[
IdentityIdentityPow:z:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�K
�
__inference__traced_save_8842
file_prefix3
/savev2_embedding_embeddings_read_readvariableop5
1savev2_embedding_1_embeddings_read_readvariableop5
1savev2_embedding_2_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop<
8savev2_adam_embedding_2_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop<
8savev2_adam_embedding_2_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
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
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2ea8f1fb79404eb88e2865b504362775/part2	
Const_1�
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop1savev2_embedding_1_embeddings_read_readvariableop1savev2_embedding_2_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop8savev2_adam_embedding_2_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop8savev2_adam_embedding_2_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�d2:(2:	�2:	�d:d:	d�:�: : : : : : : : : :	�d2:(2:	�2:	�d:d:	d�:�:	�d2:(2:	�2:	�d:d:	d�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�d2:$ 

_output_shapes

:(2:%!

_output_shapes
:	�2:%!

_output_shapes
:	�d: 

_output_shapes
:d:%!

_output_shapes
:	d�:!

_output_shapes	
:�:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�d2:$ 

_output_shapes

:(2:%!

_output_shapes
:	�2:%!

_output_shapes
:	�d: 

_output_shapes
:d:%!

_output_shapes
:	d�:!

_output_shapes	
:�:%!

_output_shapes
:	�d2:$ 

_output_shapes

:(2:%!

_output_shapes
:	�2:%!

_output_shapes
:	�d: 

_output_shapes
:d:%!

_output_shapes
:	d�:!

_output_shapes	
:�:

_output_shapes
: 
�
�
"__inference_signature_wrapper_8084
input_1
input_2
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2
*
Tout
2*(
_output_shapes
:����������*)
_read_only_resource_inputs
		*/
config_proto

GPU

CPU2 *0J 8*(
f#R!
__inference__wrapped_model_73532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������:::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :
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
: 
�
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_7584

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������d2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������d2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
?__inference_dense_layer_call_and_return_conditional_losses_7537

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAdd�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
N
2__inference_tf_op_layer_Reshape_layer_call_fn_8465

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*V
fQRO
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_74732
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������02:S O
+
_output_shapes
:���������02
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_8482

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
L__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_7457

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*
_cloned(*+
_output_shapes
:���������022
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:���������022

Identity"
identityIdentity:output:0*X
_input_shapesG
E:���������2:���������2:���������2:S O
+
_output_shapes
:���������2
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������2
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������2
 
_user_specified_nameinputs
�

E__inference_embedding_2_layer_call_and_return_conditional_losses_8432

inputs
embedding_lookup_8418
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_8418Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/8418*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/8418*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_8418*
_output_shapes
:	�2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
}
C__inference_embedding_layer_call_and_return_conditional_losses_7377

inputs
embedding_lookup_7363
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_7363Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/7363*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/7363*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_7363*
_output_shapes
:	�d2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
j
__inference_loss_fn_6_8723;
7dense_1_bias_regularizer_square_readvariableop_resource
identity��
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_1_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:�*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addc
IdentityIdentity dense_1/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�
}
C__inference_embedding_layer_call_and_return_conditional_losses_8366

inputs
embedding_lookup_8352
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_8352Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/8352*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/8352*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_8352*
_output_shapes
:	�d2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
?__inference_dense_layer_call_and_return_conditional_losses_8534

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAdd�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/addd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
 __inference__traced_restore_8944
file_prefix)
%assignvariableop_embedding_embeddings-
)assignvariableop_1_embedding_1_embeddings-
)assignvariableop_2_embedding_2_embeddings#
assignvariableop_3_dense_kernel!
assignvariableop_4_dense_bias%
!assignvariableop_5_dense_1_kernel#
assignvariableop_6_dense_1_bias 
assignvariableop_7_adam_iter"
assignvariableop_8_adam_beta_1"
assignvariableop_9_adam_beta_2"
assignvariableop_10_adam_decay*
&assignvariableop_11_adam_learning_rate
assignvariableop_12_total
assignvariableop_13_count
assignvariableop_14_total_1
assignvariableop_15_count_13
/assignvariableop_16_adam_embedding_embeddings_m5
1assignvariableop_17_adam_embedding_1_embeddings_m5
1assignvariableop_18_adam_embedding_2_embeddings_m+
'assignvariableop_19_adam_dense_kernel_m)
%assignvariableop_20_adam_dense_bias_m-
)assignvariableop_21_adam_dense_1_kernel_m+
'assignvariableop_22_adam_dense_1_bias_m3
/assignvariableop_23_adam_embedding_embeddings_v5
1assignvariableop_24_adam_embedding_1_embeddings_v5
1assignvariableop_25_adam_embedding_2_embeddings_v+
'assignvariableop_26_adam_dense_kernel_v)
%assignvariableop_27_adam_dense_bias_v-
)assignvariableop_28_adam_dense_1_kernel_v+
'assignvariableop_29_adam_dense_1_bias_v
identity_31��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp)assignvariableop_1_embedding_1_embeddingsIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp)assignvariableop_2_embedding_2_embeddingsIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_1_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_1_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0	*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_iterIdentity_7:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_1Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_2Identity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_decayIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp&assignvariableop_11_adam_learning_rateIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp/assignvariableop_16_adam_embedding_embeddings_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp1assignvariableop_17_adam_embedding_1_embeddings_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_adam_embedding_2_embeddings_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_embedding_embeddings_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp1assignvariableop_24_adam_embedding_1_embeddings_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_embedding_2_embeddings_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_kernel_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_dense_bias_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_1_kernel_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_1_bias_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_30�
Identity_31IdentityIdentity_30:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_31"#
identity_31Identity_31:output:0*�
_input_shapes|
z: ::::::::::::::::::::::::::::::2$
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
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_8623

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/adde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
t
__inference_loss_fn_2_8671E
Aembedding_2_embeddings_regularizer_square_readvariableop_resource
identity��
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_2_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	�2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/addm
IdentityIdentity*embedding_2/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�

E__inference_embedding_2_layer_call_and_return_conditional_losses_7437

inputs
embedding_lookup_7423
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_7423Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/7423*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/7423*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_7423*
_output_shapes
:	�2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
e
I__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_8549

inputs
identityS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/yj
PowPowinputsPow/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������d2
Pow[
IdentityIdentityPow:z:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
r
__inference_loss_fn_0_8645C
?embedding_embeddings_regularizer_square_readvariableop_resource
identity��
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp?embedding_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	�d2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/addk
IdentityIdentity(embedding/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
�

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_8566

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������d2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������d2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
k
1__inference_tf_op_layer_concat_layer_call_fn_8454
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*+
_output_shapes
:���������02* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_74572
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������022

Identity"
identityIdentity:output:0*X
_input_shapesG
E:���������2:���������2:���������2:U Q
+
_output_shapes
:���������2
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������2
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:���������2
"
_user_specified_name
inputs/2
�
J
.__inference_tf_op_layer_Pow_layer_call_fn_8554

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:���������d* 
_read_only_resource_inputs
 */
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_75592
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
j
__inference_loss_fn_3_8684;
7dense_kernel_regularizer_square_readvariableop_resource
identity��
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addc
IdentityIdentity dense/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ƒ
�
?__inference_model_layer_call_and_return_conditional_losses_8198
inputs_0
inputs_1
inputs_2#
embedding_embedding_lookup_8090%
!embedding_1_embedding_lookup_8096%
!embedding_2_embedding_lookup_8102(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity�s
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding/Cast�
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_8090embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/8090*+
_output_shapes
:���������2*
dtype02
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/8090*+
_output_shapes
:���������22%
#embedding/embedding_lookup/Identity�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22'
%embedding/embedding_lookup/Identity_1w
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_1/Cast�
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_8096embedding_1/Cast:y:0*
Tindices0*4
_class*
(&loc:@embedding_1/embedding_lookup/8096*+
_output_shapes
:���������2*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/8096*+
_output_shapes
:���������22'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22)
'embedding_1/embedding_lookup/Identity_1w
embedding_2/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_2/Cast�
embedding_2/embedding_lookupResourceGather!embedding_2_embedding_lookup_8102embedding_2/Cast:y:0*
Tindices0*4
_class*
(&loc:@embedding_2/embedding_lookup/8102*+
_output_shapes
:���������2*
dtype02
embedding_2/embedding_lookup�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_2/embedding_lookup/8102*+
_output_shapes
:���������22'
%embedding_2/embedding_lookup/Identity�
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22)
'embedding_2/embedding_lookup/Identity_1�
tf_op_layer_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
tf_op_layer_concat/concat/axis�
tf_op_layer_concat/concatConcatV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:00embedding_2/embedding_lookup/Identity_1:output:0'tf_op_layer_concat/concat/axis:output:0*
N*
T0*
_cloned(*+
_output_shapes
:���������022
tf_op_layer_concat/concat�
!tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����`	  2#
!tf_op_layer_Reshape/Reshape/shape�
tf_op_layer_Reshape/ReshapeReshape"tf_op_layer_concat/concat:output:0*tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*(
_output_shapes
:����������2
tf_op_layer_Reshape/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout/dropout/Const�
dropout/dropout/MulMul$tf_op_layer_Reshape/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/dropout/Mul�
dropout/dropout/ShapeShape$tf_op_layer_Reshape/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/dropout/Mul_1�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/BiasAdds
tf_op_layer_Pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
tf_op_layer_Pow/Pow/y�
tf_op_layer_Pow/PowPowdense/BiasAdd:output:0tf_op_layer_Pow/Pow/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������d2
tf_op_layer_Pow/Poww
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU�?2
dropout_1/dropout/Const�
dropout_1/dropout/MulMultf_op_layer_Pow/Pow:z:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:���������d2
dropout_1/dropout/Muly
dropout_1/dropout/ShapeShapetf_op_layer_Pow/Pow:z:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:���������d*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform�
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2"
 dropout_1/dropout/GreaterEqual/y�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������d2 
dropout_1/dropout/GreaterEqual�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������d2
dropout_1/dropout/Cast�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:���������d2
dropout_1/dropout/Mul_1�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAdd�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_embedding_lookup_8090*
_output_shapes
:	�d2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp!embedding_1_embedding_lookup_8096*
_output_shapes

:(2*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:(22+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp!embedding_2_embedding_lookup_8102*
_output_shapes
:	�2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addm
IdentityIdentitydense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������::::::::Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :
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
: 
�

E__inference_embedding_1_layer_call_and_return_conditional_losses_8399

inputs
embedding_lookup_8385
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_8385Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/8385*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/8385*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_8385*
_output_shapes

:(2*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:(22+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: 
�
�
?__inference_model_layer_call_and_return_conditional_losses_8298
inputs_0
inputs_1
inputs_2#
embedding_embedding_lookup_8204%
!embedding_1_embedding_lookup_8210%
!embedding_2_embedding_lookup_8216(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity�s
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding/Cast�
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_8204embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/8204*+
_output_shapes
:���������2*
dtype02
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/8204*+
_output_shapes
:���������22%
#embedding/embedding_lookup/Identity�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22'
%embedding/embedding_lookup/Identity_1w
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_1/Cast�
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_8210embedding_1/Cast:y:0*
Tindices0*4
_class*
(&loc:@embedding_1/embedding_lookup/8210*+
_output_shapes
:���������2*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/8210*+
_output_shapes
:���������22'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22)
'embedding_1/embedding_lookup/Identity_1w
embedding_2/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_2/Cast�
embedding_2/embedding_lookupResourceGather!embedding_2_embedding_lookup_8216embedding_2/Cast:y:0*
Tindices0*4
_class*
(&loc:@embedding_2/embedding_lookup/8216*+
_output_shapes
:���������2*
dtype02
embedding_2/embedding_lookup�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_2/embedding_lookup/8216*+
_output_shapes
:���������22'
%embedding_2/embedding_lookup/Identity�
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22)
'embedding_2/embedding_lookup/Identity_1�
tf_op_layer_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
tf_op_layer_concat/concat/axis�
tf_op_layer_concat/concatConcatV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:00embedding_2/embedding_lookup/Identity_1:output:0'tf_op_layer_concat/concat/axis:output:0*
N*
T0*
_cloned(*+
_output_shapes
:���������022
tf_op_layer_concat/concat�
!tf_op_layer_Reshape/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����`	  2#
!tf_op_layer_Reshape/Reshape/shape�
tf_op_layer_Reshape/ReshapeReshape"tf_op_layer_concat/concat:output:0*tf_op_layer_Reshape/Reshape/shape:output:0*
T0*
_cloned(*(
_output_shapes
:����������2
tf_op_layer_Reshape/Reshape�
dropout/IdentityIdentity$tf_op_layer_Reshape/Reshape:output:0*
T0*(
_output_shapes
:����������2
dropout/Identity�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/BiasAdds
tf_op_layer_Pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
tf_op_layer_Pow/Pow/y�
tf_op_layer_Pow/PowPowdense/BiasAdd:output:0tf_op_layer_Pow/Pow/y:output:0*
T0*
_cloned(*'
_output_shapes
:���������d2
tf_op_layer_Pow/Pow
dropout_1/IdentityIdentitytf_op_layer_Pow/Pow:z:0*
T0*'
_output_shapes
:���������d2
dropout_1/Identity�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAdd�
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_embedding_lookup_8204*
_output_shapes
:	�d2*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp�
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d22)
'embedding/embeddings/Regularizer/Square�
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Const�
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/Sum�
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72(
&embedding/embeddings/Regularizer/mul/x�
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mul�
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/x�
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp!embedding_1_embedding_lookup_8210*
_output_shapes

:(2*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:(22+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add�
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp!embedding_2_embedding_lookup_8216*
_output_shapes
:	�2*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�22+
)embedding_2/embeddings/Regularizer/Square�
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const�
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/Sum�
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_2/embeddings/Regularizer/mul/x�
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mul�
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x�
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x�
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add�
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense/bias/Regularizer/Square/ReadVariableOp�
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense/bias/Regularizer/Square�
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/bias/Regularizer/Const�
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/Sum�
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72
dense/bias/Regularizer/mul/x�
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/mul�
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dense/bias/Regularizer/add/x�
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/bias/Regularizer/add�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	d�2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_1/kernel/Regularizer/add/x�
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/add�
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.dense_1/bias/Regularizer/Square/ReadVariableOp�
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2!
dense_1/bias/Regularizer/Square�
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
dense_1/bias/Regularizer/Const�
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/Sum�
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72 
dense_1/bias/Regularizer/mul/x�
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/mul�
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_1/bias/Regularizer/add/x�
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_1/bias/Regularizer/addm
IdentityIdentitydense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:���������:���������:���������::::::::Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :
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
: 
�

E__inference_embedding_1_layer_call_and_return_conditional_losses_7407

inputs
embedding_lookup_7393
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_7393Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/7393*+
_output_shapes
:���������2*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/7393*+
_output_shapes
:���������22
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������22
embedding_lookup/Identity_1�
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_7393*
_output_shapes

:(2*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:(22+
)embedding_1/embeddings/Regularizer/Square�
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const�
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/Sum�
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72*
(embedding_1/embeddings/Regularizer/mul/x�
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mul�
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x�
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������22

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������
;
input_20
serving_default_input_2:0���������
;
input_30
serving_default_input_3:0���������<
dense_11
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
�b
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer_with_weights-4
layer-12
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"�^
_tf_keras_model�^{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 12885, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 40, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 263, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat", "trainable": true, "dtype": "float32", "node_def": {"name": "concat", "op": "ConcatV2", "input": ["embedding/Identity", "embedding_1/Identity", "embedding_2/Identity", "concat/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "N": {"i": "3"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"3": 1}}, "name": "tf_op_layer_concat", "inbound_nodes": [[["embedding", 0, 0, {}], ["embedding_1", 0, 0, {}], ["embedding_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["concat", "Reshape/shape"], "attr": {"Tshape": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-1, 2400]}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["tf_op_layer_concat", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow", "op": "Pow", "input": ["dense/Identity", "Pow/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 3.0}}, "name": "tf_op_layer_Pow", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["tf_op_layer_Pow", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 264, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": {"class_name": "__tuple__", "items": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0]]}, "output_layers": [["dense_1", 0, 0]]}, "build_input_shape": {"class_name": "__tuple__", "items": [{"class_name": "TensorShape", "items": [null, 18]}, {"class_name": "TensorShape", "items": [null, 18]}, {"class_name": "TensorShape", "items": [null, 12]}]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 12885, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 40, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 263, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_2", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat", "trainable": true, "dtype": "float32", "node_def": {"name": "concat", "op": "ConcatV2", "input": ["embedding/Identity", "embedding_1/Identity", "embedding_2/Identity", "concat/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "N": {"i": "3"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"3": 1}}, "name": "tf_op_layer_concat", "inbound_nodes": [[["embedding", 0, 0, {}], ["embedding_1", 0, 0, {}], ["embedding_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["concat", "Reshape/shape"], "attr": {"Tshape": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-1, 2400]}}, "name": "tf_op_layer_Reshape", "inbound_nodes": [[["tf_op_layer_concat", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["tf_op_layer_Reshape", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow", "op": "Pow", "input": ["dense/Identity", "Pow/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 3.0}}, "name": "tf_op_layer_Pow", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["tf_op_layer_Pow", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 264, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": {"class_name": "__tuple__", "items": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0]]}, "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 18]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
�

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 12885, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18]}}
�

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 40, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18]}}
�

embeddings
regularization_losses
 	variables
!trainable_variables
"	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 263, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.01, "maxval": 0.01, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
�
#regularization_losses
$	variables
%trainable_variables
&	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concat", "trainable": true, "dtype": "float32", "node_def": {"name": "concat", "op": "ConcatV2", "input": ["embedding/Identity", "embedding_1/Identity", "embedding_2/Identity", "concat/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "N": {"i": "3"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"3": 1}}}
�
'regularization_losses
(	variables
)trainable_variables
*	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Reshape", "trainable": true, "dtype": "float32", "node_def": {"name": "Reshape", "op": "Reshape", "input": ["concat", "Reshape/shape"], "attr": {"Tshape": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [-1, 2400]}}}
�
+regularization_losses
,	variables
-trainable_variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
�

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2400]}}
�
5regularization_losses
6	variables
7trainable_variables
8	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Pow", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Pow", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow", "op": "Pow", "input": ["dense/Identity", "Pow/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 3.0}}}
�
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
�

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 264, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
�
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem�m�m�/m�0m�=m�>m�v�v�v�/v�0v�=v�>v�"
	optimizer
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
Q
0
1
2
/3
04
=5
>6"
trackable_list_wrapper
Q
0
1
2
/3
04
=5
>6"
trackable_list_wrapper
�
regularization_losses
Hlayer_metrics
	variables
Imetrics
Jnon_trainable_variables
trainable_variables
Klayer_regularization_losses

Llayers
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
':%	�d22embedding/embeddings
(
�0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Mlayer_metrics
regularization_losses
	variables
Nmetrics
Onon_trainable_variables
trainable_variables
Player_regularization_losses

Qlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&(22embedding_1/embeddings
(
�0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Rlayer_metrics
regularization_losses
	variables
Smetrics
Tnon_trainable_variables
trainable_variables
Ulayer_regularization_losses

Vlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'	�22embedding_2/embeddings
(
�0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Wlayer_metrics
regularization_losses
 	variables
Xmetrics
Ynon_trainable_variables
!trainable_variables
Zlayer_regularization_losses

[layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
\layer_metrics
#regularization_losses
$	variables
]metrics
^non_trainable_variables
%trainable_variables
_layer_regularization_losses

`layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
alayer_metrics
'regularization_losses
(	variables
bmetrics
cnon_trainable_variables
)trainable_variables
dlayer_regularization_losses

elayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
flayer_metrics
+regularization_losses
,	variables
gmetrics
hnon_trainable_variables
-trainable_variables
ilayer_regularization_losses

jlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�d2dense/kernel
:d2
dense/bias
0
�0
�1"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
�
klayer_metrics
1regularization_losses
2	variables
lmetrics
mnon_trainable_variables
3trainable_variables
nlayer_regularization_losses

olayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
player_metrics
5regularization_losses
6	variables
qmetrics
rnon_trainable_variables
7trainable_variables
slayer_regularization_losses

tlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
ulayer_metrics
9regularization_losses
:	variables
vmetrics
wnon_trainable_variables
;trainable_variables
xlayer_regularization_losses

ylayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	d�2dense_1/kernel
:�2dense_1/bias
0
�0
�1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
�
zlayer_metrics
?regularization_losses
@	variables
{metrics
|non_trainable_variables
Atrainable_variables
}layer_regularization_losses

~layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
/
0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
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
0
�0
�1"
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
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
,:*	�d22Adam/embedding/embeddings/m
-:+(22Adam/embedding_1/embeddings/m
.:,	�22Adam/embedding_2/embeddings/m
$:"	�d2Adam/dense/kernel/m
:d2Adam/dense/bias/m
&:$	d�2Adam/dense_1/kernel/m
 :�2Adam/dense_1/bias/m
,:*	�d22Adam/embedding/embeddings/v
-:+(22Adam/embedding_1/embeddings/v
.:,	�22Adam/embedding_2/embeddings/v
$:"	�d2Adam/dense/kernel/v
:d2Adam/dense/bias/v
&:$	d�2Adam/dense_1/kernel/v
 :�2Adam/dense_1/bias/v
�2�
?__inference_model_layer_call_and_return_conditional_losses_7696
?__inference_model_layer_call_and_return_conditional_losses_7782
?__inference_model_layer_call_and_return_conditional_losses_8298
?__inference_model_layer_call_and_return_conditional_losses_8198�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference_model_layer_call_fn_7890
$__inference_model_layer_call_fn_8319
$__inference_model_layer_call_fn_7997
$__inference_model_layer_call_fn_8340�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_7353�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *q�n
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
�2�
C__inference_embedding_layer_call_and_return_conditional_losses_8366�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_embedding_layer_call_fn_8373�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_embedding_1_layer_call_and_return_conditional_losses_8399�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_embedding_1_layer_call_fn_8406�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_embedding_2_layer_call_and_return_conditional_losses_8432�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_embedding_2_layer_call_fn_8439�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_8447�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_tf_op_layer_concat_layer_call_fn_8454�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_8460�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
2__inference_tf_op_layer_Reshape_layer_call_fn_8465�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dropout_layer_call_and_return_conditional_losses_8477
A__inference_dropout_layer_call_and_return_conditional_losses_8482�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_dropout_layer_call_fn_8492
&__inference_dropout_layer_call_fn_8487�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_dense_layer_call_and_return_conditional_losses_8534�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_dense_layer_call_fn_8543�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_8549�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_tf_op_layer_Pow_layer_call_fn_8554�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dropout_1_layer_call_and_return_conditional_losses_8566
C__inference_dropout_1_layer_call_and_return_conditional_losses_8571�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_dropout_1_layer_call_fn_8581
(__inference_dropout_1_layer_call_fn_8576�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_dense_1_layer_call_and_return_conditional_losses_8623�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_1_layer_call_fn_8632�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_8645�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_8658�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_8671�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_8684�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_4_8697�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_5_8710�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_6_8723�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
AB?
"__inference_signature_wrapper_8084input_1input_2input_3�
__inference__wrapped_model_7353�/0=>{�x
q�n
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
� "2�/
-
dense_1"�
dense_1�����������
A__inference_dense_1_layer_call_and_return_conditional_losses_8623]=>/�,
%�"
 �
inputs���������d
� "&�#
�
0����������
� z
&__inference_dense_1_layer_call_fn_8632P=>/�,
%�"
 �
inputs���������d
� "������������
?__inference_dense_layer_call_and_return_conditional_losses_8534]/00�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� x
$__inference_dense_layer_call_fn_8543P/00�-
&�#
!�
inputs����������
� "����������d�
C__inference_dropout_1_layer_call_and_return_conditional_losses_8566\3�0
)�&
 �
inputs���������d
p
� "%�"
�
0���������d
� �
C__inference_dropout_1_layer_call_and_return_conditional_losses_8571\3�0
)�&
 �
inputs���������d
p 
� "%�"
�
0���������d
� {
(__inference_dropout_1_layer_call_fn_8576O3�0
)�&
 �
inputs���������d
p
� "����������d{
(__inference_dropout_1_layer_call_fn_8581O3�0
)�&
 �
inputs���������d
p 
� "����������d�
A__inference_dropout_layer_call_and_return_conditional_losses_8477^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
A__inference_dropout_layer_call_and_return_conditional_losses_8482^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� {
&__inference_dropout_layer_call_fn_8487Q4�1
*�'
!�
inputs����������
p
� "�����������{
&__inference_dropout_layer_call_fn_8492Q4�1
*�'
!�
inputs����������
p 
� "������������
E__inference_embedding_1_layer_call_and_return_conditional_losses_8399_/�,
%�"
 �
inputs���������
� ")�&
�
0���������2
� �
*__inference_embedding_1_layer_call_fn_8406R/�,
%�"
 �
inputs���������
� "����������2�
E__inference_embedding_2_layer_call_and_return_conditional_losses_8432_/�,
%�"
 �
inputs���������
� ")�&
�
0���������2
� �
*__inference_embedding_2_layer_call_fn_8439R/�,
%�"
 �
inputs���������
� "����������2�
C__inference_embedding_layer_call_and_return_conditional_losses_8366_/�,
%�"
 �
inputs���������
� ")�&
�
0���������2
� ~
(__inference_embedding_layer_call_fn_8373R/�,
%�"
 �
inputs���������
� "����������29
__inference_loss_fn_0_8645�

� 
� "� 9
__inference_loss_fn_1_8658�

� 
� "� 9
__inference_loss_fn_2_8671�

� 
� "� 9
__inference_loss_fn_3_8684/�

� 
� "� 9
__inference_loss_fn_4_86970�

� 
� "� 9
__inference_loss_fn_5_8710=�

� 
� "� 9
__inference_loss_fn_6_8723>�

� 
� "� �
?__inference_model_layer_call_and_return_conditional_losses_7696�/0=>���
y�v
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
p

 
� "&�#
�
0����������
� �
?__inference_model_layer_call_and_return_conditional_losses_7782�/0=>���
y�v
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
p 

 
� "&�#
�
0����������
� �
?__inference_model_layer_call_and_return_conditional_losses_8198�/0=>���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p

 
� "&�#
�
0����������
� �
?__inference_model_layer_call_and_return_conditional_losses_8298�/0=>���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p 

 
� "&�#
�
0����������
� �
$__inference_model_layer_call_fn_7890�/0=>���
y�v
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
p

 
� "������������
$__inference_model_layer_call_fn_7997�/0=>���
y�v
l�i
!�
input_1���������
!�
input_2���������
!�
input_3���������
p 

 
� "������������
$__inference_model_layer_call_fn_8319�/0=>���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p

 
� "������������
$__inference_model_layer_call_fn_8340�/0=>���
|�y
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
p 

 
� "������������
"__inference_signature_wrapper_8084�/0=>���
� 
���
,
input_1!�
input_1���������
,
input_2!�
input_2���������
,
input_3!�
input_3���������"2�/
-
dense_1"�
dense_1�����������
I__inference_tf_op_layer_Pow_layer_call_and_return_conditional_losses_8549X/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� }
.__inference_tf_op_layer_Pow_layer_call_fn_8554K/�,
%�"
 �
inputs���������d
� "����������d�
M__inference_tf_op_layer_Reshape_layer_call_and_return_conditional_losses_8460]3�0
)�&
$�!
inputs���������02
� "&�#
�
0����������
� �
2__inference_tf_op_layer_Reshape_layer_call_fn_8465P3�0
)�&
$�!
inputs���������02
� "������������
L__inference_tf_op_layer_concat_layer_call_and_return_conditional_losses_8447����
��}
{�x
&�#
inputs/0���������2
&�#
inputs/1���������2
&�#
inputs/2���������2
� ")�&
�
0���������02
� �
1__inference_tf_op_layer_concat_layer_call_fn_8454����
��}
{�x
&�#
inputs/0���������2
&�#
inputs/1���������2
&�#
inputs/2���������2
� "����������02