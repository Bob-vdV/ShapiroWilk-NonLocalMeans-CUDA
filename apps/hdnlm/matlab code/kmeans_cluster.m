function [gIdx,c,dist,clust]=kmeans_cluster(X)

    %   [gIdx, c, dist, clust] = kmeans_cluster(X) partititions the N x P data
    %   matrix X into 2 clusters, where N is the number of
    %   data points and P is the number of dimensions (variables). The
    %   partition minimizes the sum of point-to-cluster-centroid Euclidean
    %   distances of both the clusters. 
    %   gIdx is the returned N x 1 vector contains the cluster indices of each point.
    %   c is the 2 cluster centroid locations in the 2 x P matrix.
    %   dist is maximum distance of data point in the cluster 
    %   clust is the number of cluster points in the cluster
    
    [n,m]=size(X); %n = 65536 (256*256), m = 25 (pcadim)
    c=zeros(2,m); % 2 x 25
    Y=sum(X.^2,2); % 65536 x 1
    [~,minindex]=min(Y);
    [~,maxindex]=max(Y);
    c(1,:)=X(minindex,:); % 1 x 25 rows
    c(2,:)=X(maxindex,:);
    
    % allocating variables
    g0=ones(n,1); % 65536 x 1
    gIdx=zeros(n,1); % 65536 x 1
    D=zeros(n,2); % 65536 x 2
    
    % Main loop converge if previous partition is the same as current
    while any(g0~=gIdx)
        g0=gIdx;
        % Loop for each centroid
        for t=1:2
              D(:,t)=sum((X-c(t,:)).^2,2);
        end
        % Partition data to closest centroids
        [z,gIdx]=min(D,[],2); % z: 65536 x 1       Takes min column vector, stores it in z and idices in gIdx
        % Update centroids using means of partitions
        for t=1:2
            c(t,:)=mean(X(gIdx==t,:)); 
            clust(t)=size(X(gIdx==t,:),1); 
        end
    end
        for t=1:2
              dist(t)=sum(z(gIdx==t));
        end
    end
    