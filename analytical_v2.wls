#!/usr/bin/env wolframscript
(* ::Package:: *)

args = $ScriptCommandLine;
ToExpression/@Rest[args]
(*Expect: maxOrder eps sampleNumber parallel*)
R={x,y,z};
u={0,0,1};
RcrossU=R\[Cross]u;
RcrossU2=RcrossU . RcrossU;
\[Epsilon]=DiagonalMatrix[{1,1,\[Epsilon]zz}];
\[Epsilon]u=\[Epsilon]0 \[Epsilon]zz;
\[Epsilon]uni=\[Epsilon]0 \[Epsilon];
r\[Epsilon]=\[Epsilon]u R . Inverse[\[Epsilon]uni] . R//Simplify;
g\[Epsilon]uni=E^(I k0 Sqrt[r\[Epsilon]])/(4\[Pi] Sqrt[r\[Epsilon]])//Simplify;
r\[Mu]=R . R//Simplify;
g\[Mu]uni=E^(I k0 Sqrt[r\[Mu]])/(4\[Pi] Sqrt[r\[Mu]])//Simplify;
DelDel[f_]:=Table[D[f,R[[i]],R[[j]]],{i,1,3},{j,1,3}];
B=RcrossU\[TensorProduct]RcrossU/RcrossU2(\[Epsilon]u/\[Epsilon]0 g\[Epsilon]uni-g\[Mu]uni)+1/(I k0 RcrossU2)(IdentityMatrix[3]-u\[TensorProduct]u-2RcrossU\[TensorProduct]RcrossU/RcrossU2)(g\[Epsilon]uni Sqrt[r\[Epsilon]]-g\[Mu]uni Sqrt[r\[Mu]]);
GChen=-B+\[Epsilon]u Inverse[\[Epsilon]uni]g\[Epsilon]uni+1/(k0^2)DelDel[g\[Epsilon]uni];
rn=1;
(*sampleNumber=10;*)
margin=1*^-3;
\[Phi]n=Subdivide[0,2\[Pi],sampleNumber];
\[Theta]n=Subdivide[margin,\[Pi]-margin,sampleNumber/2];
k0n=2\[Pi];
\[Epsilon]zzn=eps;
numeralize[g_]:=If[parallel,Parallelize,Identity]@Outer[
Function[{\[Theta],\[Phi]},
N[g/.{y->rn Cos[\[Phi]]Sin[\[Theta]],z->rn Sin[\[Phi]]Sin[\[Theta]],x->rn Cos[\[Theta]],\[Epsilon]zz->\[Epsilon]zzn,k0->k0n}]
],
\[Theta]n,\[Phi]n
];
GChen[[;;,2]]=0;
GChen[[;;,3]]=0;
cartesians[ax_,ay_,az_]:=cartesians[ax,ay,az]=If[ax>0,
D[cartesians[ax-1,ay,az],x],
If[ay>0,
D[cartesians[ax,ay-1,az],y],
If[az>0,
D[cartesians[ax,ay,az-1],z],
GChen
]]]
Print["Computing multipole expansion..."];
Table[cartesians[ax,ay,az],
{ax,0,maxOrder},{ay,0,Clip[maxOrder-ax,{0,maxOrder}]},{az,0,Clip[maxOrder-ax-ay,{0,maxOrder}]}];
ToMultiIndex[coefficient_,powers_]:=If[powers=={},
{},
Total@(UnitVector[3,#]&/@powers)->coefficient]
CoefficientArraysToMultiIndex[array_]:=If[Length@array==1,
<|{0,0,0}->array[[1]]|>,
Association@Select[Flatten@Select[MapIndexed[ToMultiIndex,#,{Length@Dimensions@#}]&/@array,Length@#>0&]/.(index_->0):>{},Length@#>0&]];
dr=dx Sin[\[Theta]]Cos[\[Phi]]+dy Sin[\[Theta]]Sin[\[Phi]]+dz Cos[\[Theta]];
sphericalToCartesian=Table[{l,m}->4\[Pi](-I)(-1)^l/k0n^(l+1)
CoefficientArraysToMultiIndex[CoefficientArrays[(dx^2+dy^2+dz^2)^(l/2)Simplify@TrigExpand@ExpToTrig@TransformedField[
"Spherical"->"Cartesian",
SphericalHarmonicY[l,m,\[Theta],\[Phi]],
{\[Rho],\[Theta],\[Phi]}->{dx,dy,dz}
],{dx,dy,dz}]],
{l,0,maxOrder},{m,-l,l}]//Association;
Print["Computing Cartesian functions..."];
cartesiansN=Association@Flatten@Table[{ax,ay,az}->numeralize[cartesians[ax,ay,az]],
{ax,0,maxOrder},{ay,0,Clip[maxOrder-ax,{0,maxOrder}]},{az,0,Clip[maxOrder-ax-ay,{0,maxOrder}]}];
Print["Computing spherical functions..."];
sphericalsN=Association@Flatten@Table[{l,m}->Total@KeyValueMap[{\[Alpha],value}|->value cartesiansN[\[Alpha]],sphericalToCartesian[{l,m}]],
{l,0,maxOrder},{m,-l,l}];
jacobian=Table[Sin[\[Theta]n[[i\[Theta]]]],{i\[Theta],1,Length@\[Theta]n},{i\[Phi],1,Length@\[Phi]n},{i,1,3},{j,1,3}];
sphericalNorm[field_]:=Total[Abs[field]^2 jacobian,4];
optimizeCartesian[groundTruth_]:=Module[{cartesianCoeffs},
cartesianCoeffs=Flatten@Table[\[Alpha][ax,ay,az,real],
{ax,0,maxOrder},{ay,0,Clip[maxOrder-ax,{0,maxOrder}]},{az,0,Clip[maxOrder-ax-ay,{0,maxOrder}]},{real,0,1}];
getCartesianField[coeffsList_]:=Total[If[#[[1]][[4]]==0,1,I]#[[2]]cartesiansN[#[[1]][[;;3]]]&/@coeffsList,1];
getCartesianError[coeffsList_,truth_]:=
sphericalNorm[getCartesianField[coeffsList]-truth]/sphericalNorm[truth];
formatCartesianOptimizationToNormal[coeffs_]:=Map[varName|->({{#1,#2,#3,#4},varName}&@@varName),coeffs];
cartesianFunToMinimize[coeffs_]:=getCartesianError[formatCartesianOptimizationToNormal[coeffs],groundTruth];
NMinimize[cartesianFunToMinimize@cartesianCoeffs,cartesianCoeffs]
]
optimizeSpherical[groundTruth_]:=Module[{sphericalCoeffs},
sphericalCoeffs=Flatten@Table[\[Alpha][l,m,real],{l,0,maxOrder},{m,-l,l},{real,0,1}];
getSphericalField[coeffsList_]:=Total[If[#[[1]][[3]]==0,#[[2]],I #[[2]]]sphericalsN[#[[1]][[;;2]]]&/@coeffsList,1];
getSphericalError[coeffsList_,truth_]:=sphericalNorm[getSphericalField[coeffsList]-truth]/sphericalNorm[truth];
formatSphericalOptimizationToNormal[coeffs_]:=Map[varName|->({{#1,#2,#3},varName}&@@varName),coeffs];
sphericalFunToMinimize[coeffs_]:=getSphericalError[formatSphericalOptimizationToNormal[coeffs],groundTruth];
NMinimize[sphericalFunToMinimize@sphericalCoeffs,sphericalCoeffs]
]
comparison[\[Alpha]_,OptionsPattern[UseParallel->False]]:=Module[{truth,p1,p2,truthNorm,noise},
truth=cartesiansN[\[Alpha]];
If[OptionValue[UseParallel],
p1=ParallelSubmit[optimizeCartesian[truth]];
p2=ParallelSubmit[optimizeSpherical[truth]];
WaitAll[{p1,p2}],
{optimizeCartesian[truth],optimizeSpherical[truth]}
]
]
(*groundTruthSweep=ParallelTable[{ax,ay,az}->comparison[{ax,ay,az},UseParallel->False],
{ax,0,maxOrder},{ay,0,Clip[maxOrder-ax,{0,maxOrder}]},{az,0,Clip[maxOrder-ax-ay,{0,maxOrder}]}];*)
\[Alpha]GroundTruth={0,0,2};
Print["Optimizing..."];
result=comparison[\[Alpha]GroundTruth,UseParallel->parallel];
Print[Sqrt[result[[1]][[1]]]100]
Print[Sqrt[result[[2]][[1]]]100]
