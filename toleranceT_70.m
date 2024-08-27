%%%*********************************************************************************************************%%%
%% CAEA: Surrogate-assisted classification-alternative Evolutionary algorithm
%% CEC2017


clear,clc
rng('default');
rng('shuffle');
warning('off');

sn1=2;
runnum=5;       %  run time
Darr =[30];     %  dimension
for Did=1:size(Darr,2)
    d=Darr(Did);   D=d;
    if d <= 30
        maxfe=11*d;
    else
        maxfe=1000;
    end



    %% text on CEC2017
    fun_nums=30;
    fhd=str2func('cec17_func');
    targetbest = [100;200;300;400;500;600;700;800;900;1000;1100;1200;1300;1400;1500;1600;1700;1800;1900;
        2000;2100;2200;2300;2400;2500;2600;2700;2800;2900;3000];
    
    for ifun=1: fun_nums
        gfs=zeros(1,fix(maxfe/sn1));
        CE=zeros(maxfe,2);
        
        
        time_begin=tic;
        gsamp1=[];

    fname = ['tune_T70_',num2str(D),'D.txt'];
    f_out = fopen(fname,'wt');
    fprintf(f_out,'fid:%d\n',ifun);
%     fname_std = ['tune_T70_std',num2str(D),'D.txt'];
%     f_out_std = fopen(fname_std,'wt');
    fMedian = ['turnT70_fid_',num2str(ifun),'_',num2str(D),'D_locMedian.txt'];
    f_loc = fopen(fMedian,'wt'); 
    fprintf(f_loc,'fid:%d\n',ifun);

        for run=1:runnum
            fprintf('ifun: %d run: %d \n', ifun, run);
             name = ['turnT70_fid_',num2str(ifun),'_',num2str(D),'D_',num2str(run),'.dat'];
    output = fopen(name,'a');
            %---------------Initialization-----------------
            %parameter setting
            %population size
            if d <= 30
                m=5*d;
            else
                M = 100;      m = M + fix(d/10);
            end
            ps= m;


            %initialization
            p = zeros(m, d);
            v = zeros(m, d);
            %variable_domain;
            Xmin=-100;
            Xmax=100;
            varargin = [ifun,runnum];
            lu = [Xmin* ones(1, d); Xmax* ones(1, d)];
            ub = Xmax* ones(1, d);
            lb = Xmin* ones(1, d);
            Rmin = repmat(Xmin,1,D);
            Rmax = repmat(Xmax,1,D);
            FES = 0;    gen = 0;

            % minimum bounding matrix --VRmin(ps*D)
            VRmin = repmat(Rmin,ps,1);
            VRmax = repmat(Rmax,ps,1);
            XRRmin = repmat(lu(1, :), m, 1);
            XRRmax = repmat(lu(2, :), m, 1);

            %Initialize population
            p = XRRmin + (XRRmax - XRRmin) .* lhsdesign(m, d);
            %Initialize fitness
            fitness=zeros(1,m);
            for ii=1:m
                fitness(ii) = feval(fhd,p(ii,:)',varargin(1));
                FES=FES+1;
                if FES <= maxfe
                    CE(FES,:)=[FES,fitness(ii)];
                    if mod (FES,sn1)==0
                        cs1=FES/sn1;
                        gfs(1,cs1)=min(CE(1:FES,2));
                    end
                end
            end
            hx=p;   hf=fitness;
            [bestever,id] = min(fitness);
            gbpos=p(id,:);
fprintf(output,'%d\t%.15f\n',FES,bestever-targetbest(ifun));%初始化时对应的FES和fitness
            pbest = p;
            pbestval = fitness;
            [gbestval,gbestid] = min(pbestval);
            gbest = pbest(gbestid,:);
            gbestrep= repmat(gbest,ps,1);
            gbestval_old= gbestval;
            num_gen=30;  %bases on generation

            %% Main loop
            updatemodel=1;   firstupdate=1;
            tlbocount=0;   decount=0;    tlbocountsuc=0;   decountsuc=0;
            noupdate=0;   centercount=0;  subgen=0;

            if(rand>0.5)
                tlborun=1;
            else
                tlborun=0;
            end

            %% Initial parameter setting of restart strategy JADE algorithm
            F=0.5;  Cr=0.9;  Archfactor = 1.6;
            memory_size = 5;
            memory_MUF = 0.5.*ones(memory_size,1);
            memory_MUCr = 0.5.*ones(memory_size,1);
            memory_order = 1;
            decayA = [];
            T0 = 1.0;
            decayRate = T0/70;
            A = [];
            Xmax = ub;  Xmin = lb;
            counter = zeros(ps,1);
            %n = D;

            %% start loop
            while(FES < maxfe)
                % only meeting this condition, the top-ranking m data are used
                if (mod(subgen, num_gen) == 0 && subgen>1 ) || (FES == maxfe)
                    subgen=0;    updatemodel=1;

                    if(tlborun == 1)
                        tlbocount=tlbocount+1;
                    else
                        decount=decount+1;
                    end
                    tlborunbefore = tlborun;

                    [~,idx]=sort(fitness);
                    p_app=p(idx,:); f_app=fitness(idx);
                    [~,~,ip]=intersect(hx,p_app,'rows');
                    p_app(ip,:)=[];
                    f_app(ip)=[];
                    if ~isempty(p_app)==1
                        sbest_pos=p_app(1,:);

                        %top average: begin
                        tpr=rand;
                        tpc = fix(tpr*m);
                        if tpc> size (p_app,1)
                            tpc = size (p_app,1);
                        end
                        tpc =max(tpc,2) ;
                        if size (p_app,1)==1 %when only one individual remains
                            center=p_app(1,:);
                        else
                            center = mean(p_app(1:tpc,:));
                        end
                        %top average: end

                        centerfitness=FUN(center);
                        if(centerfitness<f_app(1)) %The fitness of top average is better.
                            sbest_pos=center;
                            centercount=centercount+1;
                            fprintf('centercount: %d \n', centercount );
                        end

                        sbesty = feval (fhd, sbest_pos', varargin(1));
                        FES=FES+1;
                        CE(FES,:)=[FES,sbesty];
                        if mod (FES,sn1)==0
                            cs1=FES/sn1;
                            gfs(1,cs1)=min(CE(1:FES,2));
                        end
                        hx=[hx;sbest_pos];   hf=[hf,sbesty];
                        [bestever,ib] = min([sbesty, bestever]);
                        fprintf('Iteration: %d Fitness evaluation: %d Best fitness: %e\n', gen, FES, bestever);

                        if ib==1 %get a better solution
                            gbpos=sbest_pos;
                            if(tlborun == 1)
                                tlbocountsuc=tlbocountsuc+1;
                            else
                                decountsuc=decountsuc+1;
                            end
                        else   %If no better solution is obtained, switch
                            if( tlborun == 1)
                                tlborun = 0;
                            else
                                tlborun = 1;
                            end
                        end
                    end

                    [~,idx]=sort(hf);    idx=idx(1:ps);
                    p=hx(idx,:);         fitness=hf(idx);
                    pbest = p;           pbestval = fitness;
                    [gbestval,gbestid] = min(pbestval);
                    gbest = pbest(gbestid,:);
                    gbestrep= repmat(gbest,ps,1);
                    if mod(FES-ps,floor((maxfe-ps)/20))==0
                       fprintf(output,'%d\t%.15f\n',FES,gbestval-targetbest(ifun));
                    end
                end

                if(firstupdate==1 || updatemodel==1)
                    updatemodel=0;    firstupdate = 1;

                    %% ********************* RBF modeling ************************
                    %Find the individual's NS nearest neighbors based on Euclidian distance to form the set TRAINX/Y;
                    if (d>30)
                        NS=D;
                    else
                        NS=5*D;
                    end
                    phdis=real(sqrt(p.^2*ones(size(hx'))+ones(size(p))*(hx').^2-2*p*(hx')));
                    [~,sidx]=sort(phdis,2);
                    nidx=sidx; nidx(:,NS+1:end)=[];
                    nid=unique(nidx);
                    trainx=hx(nid,:);
                    trainf=hf(nid);
                    % radial basis function interpolation----(RBF-interpolation)
                    flag='cubic';
                    [lambda, gamma]=RBF(trainx,trainf',flag);
                    FUN=@(x) RBF_eval(x,trainx,lambda,gamma,flag);
                end

                %% ********************* TLBO ************************
                if tlborun==1
                    gen = gen + 2;
                    subgen = subgen+2;
                    if subgen> num_gen
                        subgen = num_gen;
                    end
                    Partner = randperm(m);

                    for i = 1:m

                        % ----------------Begining of the Teacher Phase for ith student-------------- %
                        mean_stud = mean(p);

                        % Determination of teacher
                        [~,ind] = min(fitness);
                        best_stud = p(ind,:);

                        % Determination of the teaching factor
                        TF = randi([1 2],1,1);

                        % Generation of a new solution
                        NewSol = p(i,:) + rand(1,D).*(best_stud - TF*mean_stud);

                        % Bounding of the solution
                        NewSol = max(min(ub, NewSol),lb);

                        % Evaluation of objective function
                        %NewSolObj = FITNESSFCN(NewSol);
                        NewSolObj = FUN(NewSol);

                        % Greedy selection
                        if (NewSolObj < fitness(i))
                            p(i,:) = NewSol;
                            fitness(i) = NewSolObj;
                        end
                        % ----------------Ending of the Teacher Phase for ith student-------------- %


                        % ----------------Begining of the Learner Phase for ith student-------------- %
                        % Generation of a new solution
                        if (fitness(i)< fitness(Partner(i)))
                            NewSol = p(i,:) + rand(1, D).*(p(i,:)- p(Partner(i),:));
                        else
                            NewSol = p(i,:) + rand(1, D).*(p(Partner(i),:)- p(i,:));
                        end

                        % Bounding of the solution
                        NewSol = max(min(ub, NewSol),lb);

                        % Evaluation of objective function
                        NewSolObj =  FUN(NewSol);

                        % Greedy selection
                        if(NewSolObj< fitness(i))
                            p(i,:) = NewSol;
                            fitness(i) = NewSolObj;
                        end
                        % ----------------Ending of the Learner Phase for ith student-------------- %
                    end

                else
                    %% ********************* DE--restart strategy JADE algorithm ************************
                    gen = gen + 1;
                    subgen = subgen+1;
                    pos = p;

                    %Generating evolution matrix M.
                    Rmin = repmat(lb,1,D);   Rmax = repmat(ub,1,D);
                    fitcount = ps;

                    %% sort pbestval for mutation and crossover
                    % pbestB
                    [~,indexSel] = sort(pbestval);
                    lenSel = max(ceil(ps*rand(1,ps)),1);
                    pbestIndex = zeros(ps,1);
                    for idx = 1:ps
                        pbestIndex(idx) = indexSel(lenSel(idx));
                    end
                    pbestB = pos(pbestIndex,:);
                    % pbest posr posxr
                    rndBase = randperm(ps)';
                    psExt = ps + size(A,1);
                    rndSeq1 = ceil(rand(ps,1)*psExt);
                    for ii = 1:ps
                        while rndBase(ii)==ii
                            rndBase(ii)=ceil(rand()*ps);
                        end
                        while rndSeq1(ii)==rndBase(ii) || rndSeq1(ii)==ii
                            rndSeq1(ii) = ceil(rand()*psExt);
                        end
                    end
                    posx = [pos;A];
                    posr = pos(rndBase,:);
                    posxr = posx(rndSeq1,:);

                    %% Initial parameter setting of the memory of F and Cr
                    memory_rand_index1 = ceil(memory_size*rand(ps,1));
                    memory_rand_index2 = ceil(memory_size*rand(ps,1));
                    MUF = memory_MUF(memory_rand_index1);
                    MUCr = memory_MUCr(memory_rand_index2);
                    %for generating crossover rate Cr
                    Cr = normrnd(MUCr,0.1);
                    term_Cr = find(MUCr == -1);
                    Cr(term_Cr) = 0;
                    Cr = min(Cr,1);
                    Cr = max(Cr,0);
                    label=zeros(ps,D);
                    rndVal = rand(ps,D);
                    onemat = zeros(ps,D);
                    for ii = 1:ps
                        label(ii,:) = rndVal(ii,:)<=Cr(ii);
                        indexJ = ceil(rand()*D);
                        onemat(ii,indexJ) = 1;
                    end
                    label = label|onemat;
                    % for generating scal factor F
                    F = randCauchy(MUF,0.1);
                    term_F = find(F <= 0);
                    while ~ isempty(term_F)
                        F(term_F) = randCauchy(MUF(term_F),0.1);
                        term_F = find(F <= 0);
                    end
                    F = min(F,1);
                    FUse = repmat(F,1,D);
                    %% mutation and crossover
                    pos = pbest + FUse.*(pbestB-pbest+posr-posxr);

                    pos(~label) = pbest(~label);
                    pos = ((pos>=VRmin)&(pos<=VRmax)).*pos...
                        +(pos<VRmin).*((VRmin+pbest).*rand(ps,D)/2) ...
                        +(pos>VRmax).*((VRmax+pbest).*rand(ps,D)/2);
                    dis = (pos-pbest).*label;

                    NewSolObj = FUN(pos);
                    fitcount = fitcount + ps;
                    bin = (pbestval' > NewSolObj);   %Mark the mutation vector with better fitness;

                    %% Restart of set A
                    A = [A;pbest(bin==1,:)];
                    lengthAdd = numel(pbest(bin==1));
                    decayA = [decayA;zeros(lengthAdd,1)];
                    decayA = decayA +decayRate;

                    if(numel(decayA)>0)
                        maxDecayA = max(decayA);
                    else
                        maxDecayA = 0;
                    end
                    %Delete individuals who stayed in A for too long and didn't get better
                    if size(A,1)>round(Archfactor*ps) || maxDecayA >T0
                        MergeA = [A,decayA];
                        indexDecay = (decayA>T0);
                        MergeA(indexDecay,:) = [];
                        len = length(MergeA(:,1));
                        if len>round(Archfactor*ps)
                            rndSel = randperm(len)';
                            rndSel = rndSel(round(Archfactor*ps)+1:len);
                            MergeA(rndSel,:) = [];
                        end
                        A = MergeA(:,1:D);
                        decayA = MergeA(:,D+1);
                    end
                    %% pos and pbestval renewal
                    pbest(bin==1,:) = pos(bin==1,:);
                    pos =  pbest;
                    pbestval(bin==1) = NewSolObj(bin==1);

                    %% memory of F and Cr
                    SuccF = F(bin==1);
                    SuccCr = Cr(bin==1);
                    %dis based on the std of individual
                    dis = dis(bin==1,:);
                    dis = std(dis')';
                    dis = dis/sum(dis);
                    num_Succ = numel(SuccCr);

                    if num_Succ > 0
                        c = 0.1;
                        memory_MUF(memory_order) = (1-c)*memory_MUF(memory_order)+c*(sum(SuccF.^2))/(sum(SuccF));
                        if max(SuccCr) == 0 || memory_MUCr(memory_order) == -1
                            memory_MUCr(memory_order) = -1;
                        else
                            memory_MUCr(memory_order) = (dis'*(SuccCr.^2))/(dis'*SuccCr);
                        end
                        memory_order = memory_order + 1;
                        if memory_order > memory_size
                            memory_order = 1;
                        end
                    end

                    p = pos;
                    fitness = pbestval;
                    for i = 1:ps
                        if bin(i) == 0
                            counter(i) = counter(i) + 1;
                        else
                            counter(i) = 0;
                        end
                    end

                    %% ********************* Ending of DE--restart strategy JADE algorithm ************************
                end


            end
            fprintf(f_out,'x[%s]=%s\n',num2str(gbpos),num2str(bestever));
            bestval(run) =  bestever;
            gsamp1(run,:)=gfs;
        end
        mean_bestval = mean(bestval) ;
        fprintf(f_out,'mean-bestval=%s\n',num2str(mean_bestval - targetbest));
        time_cost=toc(time_begin);
        %         best_samp=min(gsamp1(:,end));
        %         worst_samp=max(gsamp1(:,end));
        %         samp_mean=mean(gsamp1(:,end));
        %         samp_median=median(gsamp1(:,end));
        %         gsamp1_ave=mean(gsamp1,1);
        %         gsamp1_log=log(gsamp1_ave);
        std_samp=std(gsamp1(:,end));
        fprintf(f_out,'std=%s\n',num2str(std_samp));
         [~,index] = sort(bestval);
        if mod(run,2)==0
            locMedian = [index(run/2),index(run/2+1)];
        else
            locMedian = index(ceil(run/2));
        end
         fprintf(f_loc,'LocMedian = %s\ntime(single-run-time)=%s\n',num2str(locMedian),num2str(time_cost));
        for j=1:maxfe
            if mod(j,sn1)==0
                j1=j/sn1; gener_samp1(j1)=j;
            end
        end
    end
    fclose(f_out);
%     fclose(f_out_std);
     fclose(output);
     fclose(f_loc);
end

% time_cost
function result = randCauchy(mu, sigma)
[m,n] = size(mu);
result = mu + sigma*tan(pi*(rand(m,n)-0.5));
end